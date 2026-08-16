"""
Retrieve step - CLIP-based image retrieval.

Embeds the query with CLIP's text encoder and searches the dedicated image
vector store (see pipeline.py's --multimodal, config.CHROMA_IMAGE_COLLECTION
/ QDRANT_IMAGE_COLLECTION), built from embeddings/clip_embedder.py's image
embeddings — genuine cross-modal visual similarity, independent of whatever
embedder is used for the main text store. See
generation/dual_modality_generator.py for how these results combine with
the text branch's.
"""

from embeddings.clip_embedder import ClipEmbedder
from vectorstore.base import BaseVectorStore


def retrieve_images_clip(query: str, clip_embedder: ClipEmbedder, image_store: BaseVectorStore,
                          top_k: int = 5) -> list[dict]:
    query_vec = clip_embedder.embed_texts([query])[0]
    return image_store.query(query_vec, top_k=top_k)


if __name__ == "__main__":
    print("This module is meant to be imported. See pipeline.py's --multimodal flag for a runnable example.")
