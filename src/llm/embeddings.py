from typing import Any, Dict, List, Optional

import math
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings


def get_embedding_client(
    model: str,
    api_key: str = None,
    provider: str = "vertexai",
) -> Any:
    """
    Creates an embeddings client for different providers.

    Args:
        model (str): Embedding model name (e.g., "Qwen3-Embedding-8B", "text-embedding-004", "models/embedding-001").
        base_uri (str, optional): Reserved.
        api_key (str, optional): Google API key if using google_genai; otherwise reserved.
        provider (str): "vertexai" (default), "google_genai", or "huggingface".
        device (str, optional): "cpu", "cuda", or "auto". Only used by provider "huggingface".
        pooling (str): "mean" (default) or "cls". Only used by provider "huggingface".
        max_length (int, optional): Truncation length. Only used by provider "huggingface".

    Returns:
        Any: Embeddings client instance with an embed_documents method.
    """
    provider = (provider or "vertexai").lower()
    if provider == "google_genai":
        params_gg: Dict[str, Any] = {"model": model}
        if api_key:
            params_gg["google_api_key"] = api_key
        return GoogleGenerativeAIEmbeddings(**params_gg)
    elif provider == "vertexai":
        # Default to Vertex AI text-embedding-004 if model is not provided
        model_name = model or "text-embedding-004"
        return VertexAIEmbeddings(model_name=model_name)

    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")


def embed_texts(client: Any, texts: List[str]) -> List[List[float]]:
    """
    Batch-embed texts. Uses embed_documents for consistent batching.
    """
    return client.embed_documents(texts)


def compute_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vector_a, vector_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def compute_pairwise_similarities(anchor: List[float], candidates: List[List[float]]) -> List[float]:
    return [compute_cosine_similarity(anchor, cand) for cand in candidates]



