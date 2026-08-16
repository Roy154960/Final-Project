"""
Evaluation step - RAGAS integration for faithfulness / answer relevance /
context precision / context recall, judged by a local LLM (via Ollama) so
no paid API key is required.

RAGAS defaults to OpenAI; this module rewires it to use a local Ollama
model as the judge LLM and a local HF model as the judge embedder.

Run:
    python -m evaluation.ragas_eval
"""

from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError as e:
    raise ImportError(
        "Run: pip install ragas datasets langchain-community\n"
        f"(original error: {e})"
    )


def build_local_ragas_config(ollama_llm_model: str = "llama3.2", ollama_embed_model: str = "nomic-embed-text"):
    judge_llm = LangchainLLMWrapper(ChatOllama(model=ollama_llm_model))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=ollama_embed_model))
    return judge_llm, judge_embeddings


def run_ragas_eval(samples: list[dict], ollama_llm_model: str = "llama3.2"):
    """
    samples: list of dicts with keys:
      - question: str
      - answer: str (what your pipeline generated)
      - contexts: list[str] (what was retrieved)
      - ground_truth: str (optional, needed for context_recall)
    """
    judge_llm, judge_embeddings = build_local_ragas_config(ollama_llm_model)

    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    return result


if __name__ == "__main__":
    demo_samples = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "contexts": ["Paris is the capital and largest city of France."],
            "ground_truth": "Paris",
        }
    ]
    print(run_ragas_eval(demo_samples))
