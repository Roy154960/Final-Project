"""
Groq-first, local-Ollama-fallback text generator -- the generation half
of this project's Groq integration (see groq_client.py's own module
docstring for the shared HTTP layer, and vlm/fallback_vlm.py for the
same pattern applied to vision).

Matches OllamaGenerator's own .generate()/.name interface exactly, so
this is a drop-in replacement anywhere an OllamaGenerator was
constructed directly. In this project that's exactly one call site --
mcp_server/server.py's module-level `_generator`, which every
specialist's generate_answer MCP tool call goes through (see that
module's own docstring) -- making this the single highest-traffic Groq
integration point in the whole system: every retrieval_qa, multi_hop,
painting_lookup, personal_docs, and image_qa answer that calls
generate_answer goes through here.

Every .generate() call tries Groq first (GroqGenerator, above). ANY
failure -- missing GROQ_API_KEY (GroqUnavailableError), a network error,
a rate limit, a malformed response (GroqAPIError), or any other
unexpected exception -- falls back to a local OllamaGenerator for that
SAME call, transparently, logged either way so it's obvious from the
server's own stderr which backend actually answered a given question.
Mirrors vlm/fallback_vlm.py's existing "online API first, local model
automatically after" pattern for images -- this is
the same idea for RAG text generation, just centralized in one wrapper
class instead of duplicated inline at the one call site that needs it.

Construction itself never raises, even with zero GROQ_API_KEY set --
GroqGenerator's own __init__ takes no API call at all (the missing-key
check only fires inside groq_chat_completion, at generate() time), so a
FallbackGenerator built with no Groq key configured just always takes
the Ollama branch below, with one log line per call rather than a
startup crash.

Run directly to smoke-test:
    python -m generation.fallback_generator
"""

import sys
from typing import Optional

from generation.groq_generator import GroqGenerator
from generation.ollama_generator import OllamaGenerator
from groq_client import GroqAPIError, GroqUnavailableError


class FallbackGenerator:
    def __init__(self, groq_model: Optional[str] = None, ollama_model: str = "llama3.2"):
        self._groq = GroqGenerator(groq_model) if groq_model else GroqGenerator()
        self._ollama_model = ollama_model
        self._ollama: Optional[OllamaGenerator] = None  # lazy -- see _get_ollama
        self.name = f"fallback:{self._groq.name}->ollama:{ollama_model}"

    def _get_ollama(self) -> OllamaGenerator:
        # Lazy so a session that never once needs the fallback (Groq
        # healthy the whole time) never opens a local Ollama connection
        # at all -- same "don't pay for a resource you never end up
        # using" reasoning personal_rag.py's own online-VLM-first path
        # already applies.
        if self._ollama is None:
            self._ollama = OllamaGenerator(model=self._ollama_model)
        return self._ollama

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        try:
            return self._groq.generate(question, retrieved_chunks)
        except (GroqUnavailableError, GroqAPIError) as e:
            print(
                f"[fallback_generator] Groq unavailable ({e}) -- "
                f"falling back to local Ollama ({self._ollama_model})",
                file=sys.stderr,
            )
        except Exception as e:  # noqa: BLE001 -- any other Groq-side surprise still shouldn't kill this answer
            print(
                f"[fallback_generator] Groq call failed unexpectedly ({e!r}) -- "
                f"falling back to local Ollama ({self._ollama_model})",
                file=sys.stderr,
            )
        return self._get_ollama().generate(question, retrieved_chunks)


if __name__ == "__main__":
    generator = FallbackGenerator()
    fake_chunks = [{"text": "Paris is the capital of France.", "metadata": {"filename": "geo.txt"}}]
    print(f"[{generator.name}]")
    print(generator.generate("What is the capital of France?", fake_chunks))
