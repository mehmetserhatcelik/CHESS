from typing import Any, Dict, List, Optional

import math
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
import torch
try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForTextEmbedding
except ImportError:  # pragma: no cover - older transformers versions
    from transformers import AutoModel, AutoTokenizer
    AutoModelForTextEmbedding = None


def get_embedding_client(
    model: str,
    base_uri: str = None,
    api_key: str = None,
    provider: str = "vertexai",
    device: Optional[str] = None,
    pooling: str = "mean",
    max_length: Optional[int] = None,
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
    elif provider == "huggingface":
        # Local/hosted HF model. Default device auto-detection.
        resolved_device = device
        if resolved_device is None or resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        return _HuggingFaceEmbeddingClient(
            model_name=model,
            device=resolved_device,
            pooling=pooling,
            max_length=max_length,
        )
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



class _HuggingFaceEmbeddingClient:
    """
    Minimal embedding client wrapper for HuggingFace models that exposes
    an embed_documents(texts) method to match other providers.
    """

    def __init__(self, model_name: str, device: str = "cpu", pooling: str = "mean", max_length: Optional[int] = None) -> None:
        self.model_name = model_name
        self.device = device
        self.pooling = pooling.lower() if pooling else "mean"
        self.max_length = max_length or 8192
        # Try fast tokenizer first; if it fails due to tokenizers JSON mismatch, fall back to slow
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, trust_remote_code=True
            )
        # Use AutoModelForTextEmbedding when available (e.g., Qwen3-8B-Embedding)
        if AutoModelForTextEmbedding is not None:
            try:
                self.model = AutoModelForTextEmbedding.from_pretrained(
                    model_name, trust_remote_code=True
                )
            except Exception:
                self.model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True
                )
        else:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            )
        self.model.eval()
        self.model.to(self.device)

    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return token_embeddings[:, 0]
        # mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    @torch.inference_mode()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state  # [batch, seq, hidden]
        sentence_embeddings = self._pool(token_embeddings, encoded["attention_mask"])  # [batch, hidden]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().tolist()

