"""
Smoke test for agents/contextualize.py -- fake ChatOllama, real everything
else: real prior/current-turn splitting, real transcript formatting, real
fallback-safety-net logic. Same philosophy as test_supervisor_smoke.py
(see its own docstring): no Ollama server needed, fast, catches wiring
and safety-net bugs before they cost a real model call to discover.

Run with:
    python agents/test_contextualize_smoke.py
    (or, from the project root: python -m agents.test_contextualize_smoke)
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agents import contextualize
from agents.state import AgentState


# ---------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------

class ScriptedRewriteLLM:
    """
    Stand-in for ChatOllama: returns a scripted queue of raw `.content`
    strings, one per call to ainvoke, and counts how many times it was
    actually called -- the first-turn test below asserts this stays at 0
    (no prior conversation means nothing to resolve, so no call should be
    spent). Also records every `messages` list it was called with, so
    tests can inspect exactly what the human turn contained.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        content = self._responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(content=content)


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


def _build(llm_responses):
    """Patch ChatOllama to return a ScriptedRewriteLLM, then build the
    contextualize node against it. Returns (node_fn, fake_llm) so tests
    can inspect fake_llm.call_count / fake_llm.calls afterward."""
    fake_llm = ScriptedRewriteLLM(llm_responses)
    with patch("agents.contextualize.ChatOllama", return_value=fake_llm):
        node = contextualize.build_contextualize_node()
    return node, fake_llm


def _state(messages) -> AgentState:
    return {
        "messages": messages,
        "route": None,
        "iteration_count": 0,
        "blocked": False,
        "injection_patterns": [],
    }


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

async def test_first_turn_is_a_noop_no_llm_call():
    print("\n=== a turn's first-ever message spends zero LLM calls ===")
    node, llm = _build(["should never be reached"])
    state = _state([HumanMessage(content="What is glazing in oil painting?", id="h1")])

    result = await node(state)

    _check("no state update on the first turn", result == {})
    _check("the rewrite model was never called", llm.call_count == 0)


async def test_followup_gets_rewritten_using_prior_context():
    print("\n=== a bare follow-up is rewritten using the prior turn's topic ===")
    node, llm = _build(["Which brush size is best for fine detail work?"])
    state = _state(
        [
            HumanMessage(content="What brushes work best for fine detail work?", id="h1"),
            AIMessage(content="Kolinsky sable rounds in small sizes...", name="retrieval_qa", id="a1"),
            HumanMessage(content="which size is best?", id="h2"),
        ]
    )

    result = await node(state)

    _check("exactly one replacement message returned", len(result.get("messages", [])) == 1)
    replacement = result["messages"][0]
    _check(
        "the replacement carries the rewritten, standalone question",
        replacement.content == "Which brush size is best for fine detail work?",
    )
    _check(
        "the replacement keeps the SAME message id as the original follow-up (replace, not append)",
        replacement.id == "h2",
    )

    human_turn = llm.calls[0][1].content
    _check("the prior Human turn is included in what the model saw", "What brushes work best" in human_turn)
    _check("the prior AI turn is included in what the model saw", "Kolinsky sable" in human_turn)
    _check("the current follow-up is included in what the model saw", "which size is best?" in human_turn)


async def test_already_standalone_question_is_a_noop():
    print("\n=== a question the model correctly judges already standalone is a no-op ===")
    original = "What is glazing in oil painting?"
    node, llm = _build([original])  # model echoes it back unchanged, per its own instructions
    state = _state(
        [
            HumanMessage(content="What brushes work best for detail work?", id="h1"),
            AIMessage(content="Kolinsky sable rounds...", name="retrieval_qa", id="a1"),
            HumanMessage(content=original, id="h2"),
        ]
    )

    result = await node(state)

    _check("no state update when the rewrite equals the original", result == {})
    _check("the model was still called once (it had to judge this)", llm.call_count == 1)


async def test_empty_model_output_falls_back_to_original():
    print("\n=== an empty rewrite falls back to the original message untouched ===")
    node, llm = _build([""])
    state = _state(
        [
            HumanMessage(content="What brushes work best for detail work?", id="h1"),
            AIMessage(content="Kolinsky sable rounds...", name="retrieval_qa", id="a1"),
            HumanMessage(content="which size is best?", id="h2"),
        ]
    )

    result = await node(state)

    _check("no state update on an empty model response", result == {})


async def test_implausibly_long_output_falls_back_to_original():
    print("\n=== a rewrite that looks like an ANSWER, not a rewrite, is rejected ===")
    # A real rewrite of a short follow-up adds a few words. Hundreds of
    # characters back is the model answering the question instead of
    # rewriting it -- exactly the failure this safety net guards against.
    fake_answer = (
        "Kolinsky sable brushes in sizes 0 to 2 are best for fine detail work "
        "because their fine point holds a lot of paint while still tapering "
        "to a precise tip, which is essential for controlled linework. " * 5
    )
    node, llm = _build([fake_answer])
    state = _state(
        [
            HumanMessage(content="What brushes work best for detail work?", id="h1"),
            AIMessage(content="Kolinsky sable rounds...", name="retrieval_qa", id="a1"),
            HumanMessage(content="which size is best?", id="h2"),
        ]
    )

    result = await node(state)

    _check("no state update when the model answers instead of rewriting", result == {})


async def test_context_window_caps_prior_messages_shown():
    print("\n=== only the most recent prior messages are shown to the rewrite model ===")
    node, llm = _build(["Standalone question about topic N"])
    old_messages = [
        HumanMessage(content="A question about the ancient forgotten topic zero", id="h0"),
        AIMessage(content="An answer about the ancient forgotten topic zero", name="retrieval_qa", id="a0"),
    ]
    # Enough filler turns to push the earliest ones outside the window.
    filler = []
    for i in range(1, 10):
        filler.append(HumanMessage(content=f"Question number {i}", id=f"h{i}"))
        filler.append(AIMessage(content=f"Answer number {i}", name="retrieval_qa", id=f"a{i}"))
    state = _state(old_messages + filler + [HumanMessage(content="what about that one?", id="h_last")])

    await node(state)

    human_turn = llm.calls[0][1].content
    _check(
        "the earliest, out-of-window turn is NOT shown to the model",
        "ancient forgotten topic zero" not in human_turn,
    )
    _check("a recent, in-window turn IS shown to the model", "Question number 9" in human_turn)


async def test_rewrite_that_drops_original_wording_falls_back():
    print("\n=== stricter loyalty check: a rewrite that drops/rewords an original "
          "word (not just a pronoun) falls back to the untouched original ===")
    # The model resolves the referent ("it" -> "the brushes") correctly,
    # but ALSO silently drops "cheaper" from the original message and
    # replaces it with "affordable" -- a fine paraphrase, but exactly the
    # kind of quiet reword "loyal to the original wording" exists to
    # catch and reject.
    node, llm = _build(["Are the Winsor brushes affordable for a beginner?"])
    state = _state(
        [
            HumanMessage(content="I'm looking at the Winsor brushes", id="h1"),
            AIMessage(content="Great choice for detail work.", name="product_search", id="a1"),
            HumanMessage(content="is it cheaper for a beginner?", id="h2"),
        ]
    )

    result = await node(state)

    _check(
        "the reworded rewrite is rejected -- no state update, original kept untouched",
        result == {},
    )


async def test_rewrite_that_only_resolves_referent_is_accepted():
    print("\n=== stricter loyalty check: a rewrite that ONLY substitutes the "
          "pronoun, keeping every other original word verbatim, is accepted ===")
    node, llm = _build(["is the Winsor brush cheaper for a beginner?"])
    state = _state(
        [
            HumanMessage(content="I'm looking at the Winsor brush", id="h1"),
            AIMessage(content="Great choice for detail work.", name="product_search", id="a1"),
            HumanMessage(content="is it cheaper for a beginner?", id="h2"),
        ]
    )

    result = await node(state)

    _check("exactly one replacement message returned", len(result.get("messages", [])) == 1)
    _check(
        "every original word (cheaper, beginner) survived, only the pronoun was replaced",
        result["messages"][0].content == "is the Winsor brush cheaper for a beginner?",
    )


async def test_dropped_original_words_ignores_referents_and_function_words():
    print("\n=== _dropped_original_words: pronouns and function words are exempt, "
          "content words are not ===")
    dropped = contextualize._dropped_original_words(
        "is it cheaper for a beginner?", "is the brush pricier for a beginner?"
    )
    _check("'it' (a referent word) being replaced is NOT flagged", "it" not in dropped)
    _check("'cheaper' silently becoming 'pricier' IS flagged", "cheaper" in dropped)

    clean = contextualize._dropped_original_words(
        "is it cheaper for a beginner?", "is the Winsor brush cheaper for a beginner?"
    )
    _check("a rewrite that only adds words and keeps every original one is clean", clean == [])


async def main():
    await test_first_turn_is_a_noop_no_llm_call()
    await test_followup_gets_rewritten_using_prior_context()
    await test_already_standalone_question_is_a_noop()
    await test_empty_model_output_falls_back_to_original()
    await test_implausibly_long_output_falls_back_to_original()
    await test_context_window_caps_prior_messages_shown()
    await test_rewrite_that_drops_original_wording_falls_back()
    await test_rewrite_that_only_resolves_referent_is_accepted()
    await test_dropped_original_words_ignores_referents_and_function_words()
    print("\nAll contextualize smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
