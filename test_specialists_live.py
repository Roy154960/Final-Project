"""
Live integration test for Phase 2 specialists -- real MCP server, real
Ollama, real ingested corpus. No fakes, no mocks.

This is the counterpart to test_specialists_smoke.py: that one proves the
wiring is correct cheaply and offline; this one proves the three
specialists actually produce sane answers against your real corpus before
you build the supervisor around them in Phase 3. Run this one before
moving on, not just the smoke test.

Prerequisites (same as mcp_server/README.md's):
    - ollama serve running, with the model in config.OLLAMA_GENERATION_MODELS
      pulled (e.g. `ollama pull llama3.2`).
    - A corpus already ingested into config.CHROMA_COLLECTION.
    - This file sitting at the project root, as a sibling of agents/ and
      mcp_server/ (same layout mcp_client.py already assumes).

Adjust the three QUESTION_* constants below to match your actual corpus
content -- these defaults assume art/painting-treatise material, per your
corpus, but the specific filenames and topics are guesses about what's in
there. Swap them for questions you know the answer to before trusting the
output.

Run with:
    python test_specialists_live.py
"""

import asyncio

from agents.specialists import build_specialists
from agents.state import AgentState
from langchain_core.messages import HumanMessage

# A single-topic question retrieval_qa should be able to answer directly
# from one or two retrieve() calls.
QUESTION_RETRIEVAL_QA = "What is the glazing technique in oil painting?"

# A question about the corpus itself, not its content -- corpus_meta
# should answer this from the baked-in document list alone.
QUESTION_CORPUS_META = "How many documents are in the corpus, and what are their filenames?"

# A compound question spanning two distinct sub-topics, forcing a real
# decompose -> retrieve x2 -> synthesize path in multi_hop_node.
QUESTION_MULTI_HOP = (
    "How does the tempera technique described in one treatise compare to "
    "the oil glazing technique described in another?"
)


def _print_result(label: str, state_update: dict):
    answer = state_update["messages"][-1].content
    print(f"\n=== {label} ===")
    print(answer)


async def main() -> None:
    print("Building specialists against the real MCP server + real corpus...")
    specialists = await build_specialists()

    retrieval_qa_state: AgentState = {
        "messages": [HumanMessage(content=QUESTION_RETRIEVAL_QA)],
        "route": None,
        "iteration_count": 0,
    }
    result = await specialists["retrieval_qa"](retrieval_qa_state)
    _print_result(f"retrieval_qa: {QUESTION_RETRIEVAL_QA}", result)

    corpus_meta_state: AgentState = {
        "messages": [HumanMessage(content=QUESTION_CORPUS_META)],
        "route": None,
        "iteration_count": 0,
    }
    result = await specialists["corpus_meta"](corpus_meta_state)
    _print_result(f"corpus_meta: {QUESTION_CORPUS_META}", result)

    multi_hop_state: AgentState = {
        "messages": [HumanMessage(content=QUESTION_MULTI_HOP)],
        "route": None,
        "iteration_count": 0,
    }
    result = await specialists["multi_hop"](multi_hop_state)
    _print_result(f"multi_hop: {QUESTION_MULTI_HOP}", result)

    print(
        "\nSanity-check each answer above by hand: does retrieval_qa cite a "
        "real filename from your corpus? Does corpus_meta's document list "
        "match what pipeline.py's ingestion actually reported? Does "
        "multi_hop's answer actually draw on two distinct sub-topics rather "
        "than just answering the first half? These three checks are the "
        "bar for 'ready to build the Phase 3 supervisor around this,' not "
        "just 'the script ran without an exception.'"
    )


if __name__ == "__main__":
    asyncio.run(main())
