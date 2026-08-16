"""
Common interface every vector store implements, so retrieval doesn't care
whether it's talking to ChromaDB or Qdrant.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseVectorStore(ABC):
    name: str

    @abstractmethod
    def upsert(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict]) -> None:
        ...

    @abstractmethod
    def query(self, vector: np.ndarray, top_k: int = 5, where: dict = None) -> list[dict]:
        """Return list of {id, text, score, metadata}, best first."""
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove vectors by id — used by incremental re-indexing to clean up
        stale entries for changed/deleted source files."""
        ...

    @abstractmethod
    def get_all(self) -> list[dict]:
        """Return every stored record as {id, text, metadata} (no vectors).
        Needed client-side by hybrid search (BM25 needs the whole corpus
        tokenized upfront) and by anything that wants to list what's indexed
        (e.g. the REST API's /documents endpoint)."""
        ...
