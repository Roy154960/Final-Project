"""
Prompt templates for the Generate step. Kept separate from the model
wrappers so you can iterate on prompt engineering without touching backend code.
"""

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
If the context does not contain enough information to answer, say so explicitly instead of guessing.
Always cite which piece of context you used, e.g. [source: filename, page X]."""


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.get("metadata", {}).get("filename", "unknown source")
        page = chunk.get("metadata", {}).get("page")
        label = f"{source}" + (f", page {page}" if page else "")
        context_blocks.append(f"[Context {i+1} - {label}]\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    return f"""Context:
{context_str}

Question: {question}

Answer using only the context above. Cite sources inline."""


# ---------------------------------------------------------------------------
# Dual-modality prompts (generation/dual_modality_generator.py)
# ---------------------------------------------------------------------------
# RAG_SYSTEM_PROMPT above asks the model to "say so explicitly" when it can't
# answer — fine for a human reading the output, but DualModalityGenerator
# needs to programmatically decide whether a branch's draft is usable, so it
# needs an exact, parseable token instead of free-form prose to check against.
NO_ANSWER_SENTINEL = "NO_ANSWER_IN_CONTEXT"

BRANCH_SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions using ONLY the provided context.
If the context does not contain enough information to answer, respond with EXACTLY this and nothing else:
{NO_ANSWER_SENTINEL}
Otherwise, answer normally and cite which piece of context you used, e.g. [source: filename, page X]."""

SYNTHESIS_SYSTEM_PROMPT = """You are combining up to two independent draft answers to the same question \
— one produced from retrieved text passages, one from a retrieved image — into a single final answer. \
Do not invent facts beyond what the drafts state. If the drafts agree, merge them concisely, citing both \
kinds of sources. If they disagree or cover different aspects of the question, say so explicitly rather \
than silently picking one."""


def build_synthesis_prompt(question: str, text_answer: str = None, image_answer: str = None) -> str:
    parts = [f"Question: {question}"]
    if text_answer:
        parts.append(f"\nDraft answer from retrieved text:\n{text_answer}")
    if image_answer:
        parts.append(f"\nDraft answer from a retrieved image:\n{image_answer}")
    parts.append("\nCombine the above into one final answer.")
    return "\n".join(parts)
