from sentence_transformers import SentenceTransformer
from typing import Union, List
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model
        Args:
            model_name (str): Name of the sentence-transformer model to use
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> List[float]:
        """
        Make the model callable for single text embedding
        Args:
            text: String to embed
        Returns:
            Embedding vector as list
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        Args:
            texts: List of strings to embed
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()  # Convert numpy array to list

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        Args:
            text: String to embed
        Returns:
            Embedding vector as list
        """
        embedding = self.model.encode(text)
        return embedding.tolist()  # Convert numpy array to list

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        """
        embedding1 = self.model.encode(text1)
        embedding2 = self.model.encode(text2)
        return self._cosine_similarity(embedding1, embedding2)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        """
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return dot_product / (norm1 * norm2)


