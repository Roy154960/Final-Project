"""
Retrieve step - re-ranking.
Initial retrieval (vector or hybrid) is optimized for recall over a large
candidate set; a cross-encoder reranker then re-scores the top candidates
for precision, since it looks at the query and chunk together instead of
comparing independent embeddings. Free, local, no API key.
"""

# Guarded the same way embeddings/hf_embedder.py already guards its own
# sentence_transformers import -- and for the same confirmed reason: a
# broken/mismatched torch+torchvision install doesn't raise a plain
# ImportError here, it raises transformers' own lazy-import wrapper
# (ModuleNotFoundError, itself caused by a RuntimeError -- "operator
# torchvision::nms does not exist" -- deep in transformers.modeling_utils).
# ModuleNotFoundError IS an ImportError subclass, so this still catches it.
# Before this guard existed, this bare top-level import crashed the whole
# mcp-server process (server.py's `from retrieval.reranker import Reranker`
# runs before FastMCP binds a port and before the try/except server.py
# already has around HFEmbedder()/Reranker() *construction* -- that guard
# never got a chance to run because the failure happened one level up, at
# import time, not construction time).
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from config import HF_RERANKER_MODEL


class Reranker:
    def __init__(self, model_name: str = HF_RERANKER_MODEL):
        if CrossEncoder is None:
            raise ImportError("Run: pip install sentence-transformers")
        self.name = f"reranker:{model_name}"
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]


if __name__ == "__main__":
    reranker = Reranker()
    candidates = [
        {"id": "1", "text": "Paris is the capital of France."},
        {"id": "2", "text": "The Eiffel Tower is a famous landmark in Paris."},
        {"id": "3", "text": "Bananas are a good source of potassium."},
    ]
    result = reranker.rerank("What is the capital of France?", candidates)
    for r in result:
        print(f"{r['rerank_score']:.3f}  {r['text']}")
