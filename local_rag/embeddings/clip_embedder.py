"""
Embed step - CLIP via open_clip (free, local, open-source LAION weights).

This is the true multimodal embedder: images and text are projected into
the SAME vector space, so a text query like "a red sports car" can retrieve
an image chunk directly by cosine similarity, with no OCR or captioning
step in between.

Run directly to smoke-test:
    python -m embeddings.clip_embedder
"""

import gc
import numpy as np
from pathlib import Path
from PIL import Image
from embeddings.base import BaseEmbedder
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED

# Images are embedded in fixed-size batches instead of all at once. CLIP's
# forward-pass activations scale with batch size, and a heavily-illustrated
# book can easily hand embed_images() a couple hundred images in one call —
# on CPU that's a genuine multi-GB RAM spike. Batching bounds memory use to
# roughly this many images' worth of activations regardless of book size.
CLIP_IMAGE_BATCH_SIZE = 16
from utils.image_sniff import describe_ext_mismatch

try:
    import torch
    import open_clip
except ImportError:
    torch = None
    open_clip = None

CLIP_DIMS = {
    "ViT-B-32": 512,
    "ViT-L-14": 768,
}


class ClipEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED):
        if open_clip is None:
            raise ImportError("Run: pip install open-clip-torch torch")
        self.name = f"clip:{model_name}/{pretrained}"
        self.dimensions = CLIP_DIMS.get(model_name, 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

    def supports_images(self) -> bool:
        return True

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self._tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        tensors = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(self._preprocess(img))
                valid_indices.append(i)
            except Exception as e:
                # A single corrupted/unreadable image (seen in practice with
                # some PDF-extracted JPEG2000 images) used to crash this
                # whole batch, losing every other image's embedding too and
                # taking the entire /ingest request down with it — even
                # after text chunks had already been embedded and stored.
                # Skipped images get a zero vector instead: embed_images()'s
                # 1:1 paths-in/vectors-out contract stays exactly the same
                # for every caller (no signature change needed anywhere),
                # and a zero vector is inert at retrieval time — cosine
                # similarity against it is 0, so it's never surfaced as a
                # match.
                diagnosis = describe_ext_mismatch(path, Path(path).suffix)
                if diagnosis:
                    print(f"[warn] failed to embed image {path}: {e}\n"
                          f"       -> likely cause: {diagnosis}\n"
                          f"       -> if this is JPEG2000, install openjpeg support for Pillow "
                          f"(pip install --upgrade pillow — most prebuilt wheels already include it; "
                          f"if it still fails, your Pillow build was compiled without OpenJPEG) — "
                          f"otherwise the source PDF's image stream for this one is likely truncated/"
                          f"corrupted and there's no fix short of re-extracting from a cleaner source PDF")
                else:
                    print(f"[warn] failed to embed image {path}: {e}")

        features = np.zeros((len(image_paths), self.dimensions), dtype=np.float32)
        if tensors:
            # Loop in fixed-size batches instead of stacking every image from
            # the whole book into one forward pass. Bounds peak memory to one
            # batch's worth of activations no matter how many images come in.
            for start in range(0, len(tensors), CLIP_IMAGE_BATCH_SIZE):
                chunk_tensors = tensors[start:start + CLIP_IMAGE_BATCH_SIZE]
                chunk_indices = valid_indices[start:start + CLIP_IMAGE_BATCH_SIZE]

                batch = torch.stack(chunk_tensors).to(self.device)
                with torch.no_grad():
                    encoded = self._model.encode_image(batch)
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                features[chunk_indices] = encoded.cpu().numpy().astype(np.float32)

                # Drop references to this batch's tensors/activations and
                # clear the allocator cache before starting the next batch,
                # so memory doesn't creep up over the course of a big run.
                del batch, encoded
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                else:
                    # No CUDA cache to clear on CPU — force garbage collection
                    # instead so the batch's activations are actually freed
                    # before the next one is allocated, rather than lingering
                    # until Python gets around to it on its own schedule.
                    gc.collect()
        return features


if __name__ == "__main__":
    embedder = ClipEmbedder()
    text_vecs = embedder.embed_texts(["a diagram of a RAG pipeline"])
    print(f"{embedder.name} text: shape={text_vecs.shape}")
    print("Pass a real image path to embed_images([...]) to test image embedding.")
