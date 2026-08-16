"""
Groq-first, local-Ollama-fallback VLM -- vision half of this project's
Groq integration (see groq_client.py's own module docstring for the
shared HTTP layer, and generation/fallback_generator.py for the same
pattern applied to RAG text generation).

Same describe_image()/answer_with_image() interface as every other VLM
backend in vlm/ (OllamaVLM, HFVLM, GroqVLM) -- see
ingestion/image_captioning.py's load_vlm() for where this is wired in as
the "groq" backend choice (config.py's PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND's
new default), and mcp_server/image_tools.py's own _load_vlm() for the
other call site that uses this directly (its live-caption fallback path
for a corpus image with no stored caption -- see that module's own
docstring).

Every call tries GroqVLM first. ANY failure -- missing GROQ_API_KEY
(GroqUnavailableError), a network error, a rate limit, a malformed
response (GroqAPIError), or any other unexpected exception -- falls back
to a local OllamaVLM for that SAME call, logged either way so it's
obvious from the server's own stderr which backend actually answered.

Construction itself never raises, even with zero GROQ_API_KEY set --
GroqVLM's own __init__ takes no API call at all (the missing-key check
only fires at describe_image()/answer_with_image() time), so a
FallbackVLM built with no Groq key configured just always takes the
Ollama branch below, with one log line per call rather than a startup
crash or an exception the caller has to know to catch just to construct
this class.

Run directly to smoke-test:
    python -m vlm.fallback_vlm data/raw/sample.png "What is shown in this image?"
"""

import sys
from pathlib import Path
from typing import Optional

from groq_client import GroqAPIError, GroqUnavailableError
from vlm.groq_vlm import GroqVLM


class FallbackVLM:
    def __init__(self, groq_model: Optional[str] = None, ollama_model: Optional[str] = None):
        self._groq = GroqVLM(groq_model) if groq_model else GroqVLM()
        self._ollama_model = ollama_model
        self._ollama = None  # lazy -- see _get_ollama; type is vlm.ollama_vlm.OllamaVLM
        self.name = f"fallback:{self._groq.name}->ollama"

    def _get_ollama(self):
        # Imported lazily (not at module top) for the same reason
        # ingestion/image_captioning.py's own load_vlm() already imports
        # each backend inside its own branch rather than all up front --
        # a caller that never actually needs the Ollama fallback this
        # session shouldn't pay for importing/loading it at all.
        if self._ollama is None:
            from vlm.ollama_vlm import OllamaVLM

            self._ollama = OllamaVLM(self._ollama_model) if self._ollama_model else OllamaVLM()
        return self._ollama

    def _with_fallback(self, method_name: str, *args, **kwargs) -> str:
        try:
            return getattr(self._groq, method_name)(*args, **kwargs)
        except (GroqUnavailableError, GroqAPIError) as e:
            print(
                f"[fallback_vlm] Groq unavailable ({e}) -- falling back to local Ollama VLM",
                file=sys.stderr,
            )
        except Exception as e:  # noqa: BLE001 -- any other Groq-side surprise still shouldn't drop this image
            print(
                f"[fallback_vlm] Groq vision call failed unexpectedly ({e!r}) -- "
                "falling back to local Ollama VLM",
                file=sys.stderr,
            )
        return getattr(self._get_ollama(), method_name)(*args, **kwargs)

    def describe_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        return self._with_fallback("describe_image", image_path, prompt)

    def answer_with_image(self, question: str, image_path: str, text_context: str = "") -> str:
        return self._with_fallback("answer_with_image", question, image_path, text_context)


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is shown in this image?"

    if not Path(image_path).exists():
        print(f"No image found at {image_path} — pass a real image path to test this.")
    else:
        vlm = FallbackVLM()
        print(f"[{vlm.name}]")
        print(vlm.answer_with_image(question, image_path))
