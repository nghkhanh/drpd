from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
import requests
from drpd.config import app_config

class BaseEmbedding(ABC):
    """Abstract base class for embedding managers."""
    @abstractmethod
    def embed(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Convert text into embedding."""
        pass


# TODO
# Local LLM

class EmbeddingModel(BaseEmbedding):
    """Embedding using remote url."""
    def __init__(self):
        """Initialize the embedding manager."""
        self.url = app_config["embedding"]["url"]
        self.model = app_config["embedding"]["model"]
    
    def embed(self, texts, batch_size = 32) -> np.ndarray:
        """Generate embedding for input text."""
        if isinstance(texts, str):
            texts = [texts]
        
        import concurrent.futures

        def _process_batch(batch_texts):
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
                    }
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as e:
                raise RuntimeError(f"Embedding failed for batch {e}") from e
        
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(_process_batch, batches))

        flat_embeddings = [emb for batch_result in results for emb in batch_result]

        return np.array(flat_embeddings)
        
encoder = EmbeddingModel()