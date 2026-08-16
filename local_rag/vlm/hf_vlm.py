"""
VLM method 2: vision-language model via local Hugging Face weights.

Uses moondream2 by default — a genuinely small VLM (under 2B params) built
specifically to run fast on CPU, which matters for a RAG pipeline where you
don't want a 10+ second wait per image question. Qwen2-VL-2B-Instruct is
included as a stronger-but-slower alternative in the same file so you can
compare both without juggling two separate scripts.

Run directly to smoke-test:
    python -m vlm.hf_vlm data/raw/sample.png "What is shown in this image?"
"""

import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from PIL import Image
except ImportError:
    torch = None

MODEL_OPTIONS = {
    "moondream2": "vikhyatk/moondream2",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
}


class HFVLM:
    def __init__(self, model_key: str = "moondream2"):
        if torch is None:
            raise ImportError("Run: pip install transformers torch pillow")
        if model_key not in MODEL_OPTIONS:
            raise ValueError(f"Unknown model_key '{model_key}'. Options: {list(MODEL_OPTIONS)}")

        model_name = MODEL_OPTIONS[model_key]
        self.name = f"hf-vlm:{model_name}"
        self.model_key = model_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_key == "moondream2":
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:  # qwen2-vl
            from transformers import Qwen2VLForConditionalGeneration
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._processor = AutoProcessor.from_pretrained(model_name)

    def answer_with_image(self, question: str, image_path: str, text_context: str = "") -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = f"{(text_context + ' ') if text_context else ''}{question}"

        if self.model_key == "moondream2":
            encoded_image = self._model.encode_image(image)
            return self._model.answer_question(encoded_image, prompt, self._tokenizer)

        else:  # qwen2-vl
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(text=[text_prompt], images=[image], return_tensors="pt").to(self.device)
            output = self._model.generate(**inputs, max_new_tokens=256)
            return self._processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is shown in this image?"

    if not Path(image_path).exists():
        print(f"No image found at {image_path} — pass a real image path to test this.")
    else:
        vlm = HFVLM("moondream2")
        print(vlm.answer_with_image(question, image_path))
