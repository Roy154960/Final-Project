"""
VLM method 4: vision-language model via Groq's hosted FREE tier (not
local). Vision half of this project's Groq integration -- see
groq_client.py's own module docstring for the shared HTTP layer, and
generation/groq_generator.py for the text/RAG-generation half.

Same describe_image()/answer_with_image() interface as vlm/ollama_vlm.py's
OllamaVLM and vlm/hf_vlm.py's HFVLM -- so ingestion/image_captioning.py's
load_vlm() and every
existing caller that already knows how to swap VLM backends can use this
one the same way. Used ONLY through vlm/fallback_vlm.py's FallbackVLM in
practice -- see that module's own docstring for why nothing in this
project should construct GroqVLM directly and skip the automatic local
fallback it provides.

Model: qwen/qwen3.6-27b -- Groq's own multimodal (text + vision) model
as of the check against https://console.groq.com/docs/vision (2026-08).
Verify against that page if this stops matching what Groq's docs/
dashboard list; hosted-platform model names change without much notice
(config.py's own GROQ_VISION_MODEL constant is the one place to update
if so).

CONFIRMED live problem, now fixed: qwen/qwen3.6-27b is a dual-mode
reasoning model (see https://console.groq.com/docs/model/qwen/qwen3.6-27b)
that, left at its default reasoning mode, prepends a whole
<think>...</think> block to every single response -- a live captioning
run showed that block landing VERBATIM in what was supposed to be a
short image caption, and burning through the free tier's per-minute
token budget several times faster than a direct answer would (a
reasoning preamble alone can run into the hundreds of tokens before the
model even starts the real caption). Every call below now passes
reasoning_effort="none" (see groq_client.py's own comment on that
param for why this, not reasoning_format="hidden", is the fix that
actually saves tokens rather than just hiding them) -- a plain
"describe this image" caption has no real use for step-by-step
reasoning anyway. _generate() also strips any <think>...</think> block
that slips through regardless (a different Groq vision model reached
via config.py's GROQ_VISION_MODEL might not honor reasoning_effort the
same way, or might not support the qwen3 family's disable path at all)
so a caption is never accidentally stored as raw chain-of-thought again
even if the upstream fix stops applying for some future model swap.

Uses base64-data-URI image encoding, sent as an OpenAI-shaped
`image_url` content part per
https://console.groq.com/docs/vision#how-to-pass-locally-saved-images-as-input.

Never silently no-ops: a missing GROQ_API_KEY surfaces as
groq_client.GroqUnavailableError with the exact fix, same "tell the
person running this the real fix" convention config.py's TESSERACT_CMD
comment already follows. Every
caller of this class is expected to catch that (and GroqAPIError) and
fall back to a local VLM -- see vlm/fallback_vlm.py.

Run directly to smoke-test:
    python -m vlm.groq_vlm data/raw/sample.png "What is shown in this image?"
"""

import base64
import mimetypes
import re
import sys
from pathlib import Path

from config import GROQ_VISION_MODEL
from groq_client import groq_chat_completion, GROQ_VISION_REQUEST_TIMEOUT_SECONDS

# Strips a <think>...</think> block (including the tags themselves) from
# a model response -- see this module's own top docstring for exactly
# why this exists as a defensive backstop even though reasoning_effort=
# "none" below should mean it never has anything to strip in practice.
# re.DOTALL so the block can span multiple lines (it always does in
# practice -- reasoning is rarely one line).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class GroqVLM:
    def __init__(self, model: str = GROQ_VISION_MODEL):
        self.name = f"groq-vlm:{model}"
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"  # Groq needs SOME image/* mime type; a safe default
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{data}"

    def _generate(self, prompt: str, image_path: str) -> str:
        data_uri = self._encode_image(image_path)
        data = groq_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            model=self.model,
            node="vision",
            # See this module's own top docstring -- a caption has no
            # real use for step-by-step reasoning, and leaving this
            # unset let a <think> block both corrupt captions and burn
            # through the rate limit several times faster than needed.
            reasoning_effort="none",
            # A vision payload (full base64 image, not a few words of
            # text) needs longer than groq_chat_completion's default
            # text-completion timeout to finish uploading, especially
            # over a slower connection -- see groq_client.py's own
            # GROQ_VISION_REQUEST_TIMEOUT_SECONDS comment for the
            # confirmed live TimeoutError this fixes.
            timeout=GROQ_VISION_REQUEST_TIMEOUT_SECONDS,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            # A blocked/empty response still comes back as valid JSON,
            # just without the usual shape -- degrade to a plain, honest
            # string rather than raising on a response that technically
            # succeeded at the HTTP layer.
            return "(the online VLM returned an empty response for this image)"
        content = _THINK_BLOCK.sub("", content or "").strip()
        return content or "(the online VLM returned an empty response for this image)"

    def describe_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        return self._generate(prompt, image_path)

    def answer_with_image(self, question: str, image_path: str, text_context: str = "") -> str:
        """
        Same contract as OllamaVLM.answer_with_image: combine a
        retrieved/uploaded image with any retrieved text context and the
        user's question in one VLM call.
        """
        prompt = f"{('Context: ' + text_context + chr(10) + chr(10)) if text_context else ''}Question: {question}"
        return self._generate(prompt, image_path)


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is shown in this image?"

    if not Path(image_path).exists():
        print(f"No image found at {image_path} — pass a real image path to test this.")
    else:
        vlm = GroqVLM()
        print(f"[{vlm.name}]")
        print(vlm.answer_with_image(question, image_path))
