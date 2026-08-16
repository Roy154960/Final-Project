"""
Generate step - dual-modality generation.

Pairs with pipeline.py's --multimodal flag: text chunks live in their own
vector store (embedded by whichever --embedder was chosen: hf/ollama),
images live in a separate CLIP-embedded store (config.CHROMA_IMAGE_COLLECTION
/ QDRANT_IMAGE_COLLECTION), and each image's VLM-written caption
(ingestion/image_captioning.py) is ALSO dual-indexed into the text store so
images are reachable by plain text search too, not only by CLIP's own
cross-modal similarity. See retrieval/image_retriever.py for the image-side
query path.

At generation time each branch is drafted independently — a text LLM call
grounded in retrieved text/caption chunks, a VLM call grounded in the
retrieved image itself — then, if both drafts are viable, one more LLM call
synthesizes them into a single final answer. This lets each branch use the
generation approach suited to it (a VLM actually looks at the image; a text
LLM reasons over passages) instead of forcing both through one combined
prompt.

Abstention: a branch never contributes to the final answer unless it passes
two checks, cheapest first:
  1. Its top retrieval score clears config.TEXT_RELEVANCE_SCORE_THRESHOLD /
     IMAGE_RELEVANCE_SCORE_THRESHOLD — filters obviously-irrelevant retrieval
     before spending any LLM/VLM call on it.
  2. Its own draft-generation call is instructed to reply with
     prompts.NO_ANSWER_SENTINEL instead of guessing if, having actually seen
     the retrieved content, it still doesn't answer the question — the
     model's own judgment, made in the same call that would otherwise
     produce the draft (no separate classification call needed).
If neither branch survives both checks, the final answer is a fixed
"not enough information" message and no synthesis call is made either, so
nothing gets invented at that stage.

Known limitation: the sentinel check only works as well as the underlying
model's instruction-following — a model that ignores the "reply with
exactly X" instruction and answers anyway will look "viable" even on thin
context. This is a heuristic layered on top of retrieval scores, not a
guarantee, same caveat this project already documents for the prompt-
injection pattern scanner in safety/prompt_injection.py.
"""

from config import TEXT_RELEVANCE_SCORE_THRESHOLD, IMAGE_RELEVANCE_SCORE_THRESHOLD
from generation.prompts import (
    build_rag_prompt, BRANCH_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT,
    NO_ANSWER_SENTINEL, build_synthesis_prompt,
)

NO_INFO_MESSAGE = (
    "I don't have enough information in the indexed documents or images to answer this question."
)


class DualModalityGenerator:
    def __init__(self, text_generator, vlm_backend: str = "ollama", vlm_model: str = None):
        """
        text_generator: an OllamaGenerator / HFGenerator / VLLMServerGenerator
            instance — NOT MultimodalGenerator ("vlm"), which requires an
            image chunk and raises on text-only input, breaking the text
            branch here. Reused for both the text branch's draft answer and
            the final synthesis call.
        vlm_backend/vlm_model: which VLM answers the image branch — see vlm/.
        """
        if vlm_backend == "ollama":
            from vlm.ollama_vlm import OllamaVLM
            self._vlm = OllamaVLM(vlm_model or "llava")
        elif vlm_backend == "hf":
            from vlm.hf_vlm import HFVLM
            self._vlm = HFVLM(vlm_model or "moondream2")
        else:
            raise ValueError(f"Unknown vlm_backend: {vlm_backend}")
        self.text_generator = text_generator
        self.name = f"dual:{text_generator.name}+{self._vlm.name}"

    def _call_text_generator(self, prompt: str, system_prompt: str) -> str:
        gen = self.text_generator
        if hasattr(gen, "client"):  # OllamaGenerator — raw .chat() lets us swap the system prompt directly
            response = gen.client.chat(model=gen.model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ])
            return response["message"]["content"]
        # HFGenerator / VLLMServerGenerator: no raw prompt hook, so reuse .generate()
        # with the prompt as the "question" and no extra context — the same trick
        # retrieval/multi_query.py and retrieval/contextual_compression.py already
        # use for this exact problem — passing system_prompt through (both now
        # accept it) so the sentinel instruction actually applies for these
        # backends too, not just Ollama's.
        return gen.generate(prompt, [], system_prompt=system_prompt)

    def _text_branch(self, question: str, text_chunks: list[dict]):
        if not text_chunks:
            return None
        top_score = max((c.get("score", 0.0) for c in text_chunks), default=0.0)
        if top_score < TEXT_RELEVANCE_SCORE_THRESHOLD:
            return None
        prompt = build_rag_prompt(question, text_chunks)
        raw = self._call_text_generator(prompt, BRANCH_SYSTEM_PROMPT).strip()
        if raw == NO_ANSWER_SENTINEL:
            return None
        return raw

    def _image_branch(self, question: str, image_chunks: list[dict]):
        if not image_chunks:
            return None
        top = max(image_chunks, key=lambda c: c.get("score", 0.0))
        if top.get("score", 0.0) < IMAGE_RELEVANCE_SCORE_THRESHOLD:
            return None
        image_path = top.get("metadata", {}).get("image_path")
        if not image_path:
            return None
        caption = top.get("metadata", {}).get("caption", "")
        prompted_question = (
            f"{question}\n\nIf the image does not actually help answer the question, "
            f"reply with exactly: {NO_ANSWER_SENTINEL}"
        )
        raw = self._vlm.answer_with_image(prompted_question, image_path, text_context=caption).strip()
        if raw == NO_ANSWER_SENTINEL:
            return None
        return raw

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """
        retrieved_chunks: the concatenation of the text-store retrieval
        results and the image-store retrieval results (see pipeline.py
        cmd_ask), each tagged metadata["retrieval_branch"] = "text" | "image"
        so they can be split back apart here.
        """
        text_chunks = [c for c in retrieved_chunks if c.get("metadata", {}).get("retrieval_branch") != "image"]
        image_chunks = [c for c in retrieved_chunks if c.get("metadata", {}).get("retrieval_branch") == "image"]

        text_answer = self._text_branch(question, text_chunks)
        image_answer = self._image_branch(question, image_chunks)

        if text_answer is None and image_answer is None:
            return NO_INFO_MESSAGE
        if image_answer is None:
            return text_answer
        if text_answer is None:
            return image_answer

        # Both viable — synthesize one final answer. Only reached when there's
        # genuinely something from each side to reconcile; a single viable
        # branch is returned directly above rather than paraphrased for no reason.
        synthesis_prompt = build_synthesis_prompt(question, text_answer, image_answer)
        return self._call_text_generator(synthesis_prompt, SYNTHESIS_SYSTEM_PROMPT).strip()


if __name__ == "__main__":
    print("This module needs a live text generator + VLM backend. "
          "See pipeline.py's --multimodal flag for a runnable end-to-end example.")
