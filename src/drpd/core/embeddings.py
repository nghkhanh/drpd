from abc import ABC, abstractmethod

import numpy as np
import requests

from drpd.config import app_config


class BaseEmbedding(ABC):
    """Abstract base class for embedding managers."""

    @abstractmethod
    def embed(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Convert text into embedding."""
        pass


# TODO(khanhnh): support local embedding.  # noqa: TD003
# Local LLM


class EmbeddingModel(BaseEmbedding):
    """Embedding using remote url (OpenAI-compatible API)."""

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        """
        Initialize the embedding manager.
        Priority: Arguments passed -> Config file -> Error
        """
        self.url: str = url or str(app_config.get("url") or "")
        self.model: str = model or str(app_config.get("model") or "")

        if not self.url:
            raise ValueError("Embedding URL is not configured.")

    def embed(self, texts: str | list[str], batch_size=32, **kwargs) -> np.ndarray:
        """Generate embedding for input text."""
        if isinstance(texts, str):
            texts = [texts]

        import concurrent.futures

        def _process_batch(batch_texts: list[str]) -> list[list[float]]:
            try:
                response = requests.post(
                    self.url,
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": batch_texts,
                    },
                )
                response.raise_for_status()
                # Cung cấp kiểu dữ liệu cụ thể hơn cho 'embedding'
                data: dict[str, list[dict[str, list[float]]]] = response.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as e:
                raise RuntimeError(f"Embedding failed for batch {e}") from e

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results: list[list[list[int | float]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results: list[list[list[int | float]]] = list(
                executor.map(_process_batch, batches)
            )

        flat_embeddings: list[list[int | float]] = [
            emb for batch_result in results for emb in batch_result
        ]

        return np.array(flat_embeddings)


encoder = EmbeddingModel()
