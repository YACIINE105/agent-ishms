import cohere
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Remote Cohere Rerank client (rerank-v3.5).
    Operates without loading PyTorch or SentenceTransformers into worker RAM.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-v3.5"):
        self.api_key = api_key
        self.model = model
        self.client = cohere.Client(api_key=api_key) if api_key else None

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Tuple[int, float]]:
        if not documents:
            return []

        if not self.client:
            return [(idx, 1.0) for idx in range(min(len(documents), top_n))]

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents))
            )
            return [(result.index, float(result.relevance_score)) for result in response.results]
        except Exception as e:
            logger.error(f"Remote Cohere rerank call failed: {e}")
            return [(idx, 1.0) for idx in range(min(len(documents), top_n))]
