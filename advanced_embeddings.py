"""
Advanced Embedding Engine for LEO Optima
=========================================
Provides semantic embeddings using flexible interfaces (OpenAI, Local, etc.)
Model Agnostic: Works with any embedding model provider.
"""

import numpy as np
import os
import hashlib
import json
from typing import Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

class AdvancedEmbeddingEngine:
    """
    Production-grade semantic embedding engine with multiple backends:
    1. Provided Embedding Interface (OpenAI, custom API, etc.)
    2. Sentence-Transformers (Local)
    3. Johnson-Lindenstrauss projection (Fallback)
    """
    
    def __init__(
        self, 
        dim: int = 384,
        model_interface: Any = None,
        use_local_transformer: bool = True,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "leo_storage/embeddings_cache"
    ):
        self.dim = dim
        self.model_interface = model_interface
        self.use_local_transformer = use_local_transformer and SENTENCE_TRANSFORMERS_AVAILABLE
        self.cache_dir = cache_dir
        self.model_name = model_name
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize sentence-transformer if enabled and interface is not provided
        self.transformer_model = None
        if self.use_local_transformer and not self.model_interface:
            try:
                print(f"🔄 Loading sentence-transformer model: {model_name}...")
                self.transformer_model = SentenceTransformer(model_name)
                test_emb = self.transformer_model.encode("test")
                self.dim = len(test_emb)
                print(f"✅ Sentence-Transformer initialized (dim={self.dim})")
            except Exception as e:
                print(f"⚠️ Sentence-Transformer failed: {e}. Falling back to projection.")
                self.use_local_transformer = False
        
        # Initialize fallback projection matrix
        if not self.use_local_transformer and not self.model_interface:
            state = np.random.RandomState(42)
            self.projection = state.randn(1024, self.dim).astype(np.float32)
            self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)
            print(f"⚠️ Using fallback Johnson-Lindenstrauss projection (dim={self.dim})")

    def _get_cache_path(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.npy")

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        if use_cache:
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                try:
                    return np.load(cache_path)
                except:
                    pass

        # Try Provided Interface (OpenAI, custom, etc.)
        if self.model_interface:
            try:
                # Assuming interface has an 'embed' or 'encode' method
                if hasattr(self.model_interface, 'embed'):
                    embedding = self.model_interface.embed(text)
                elif hasattr(self.model_interface, 'encode'):
                    embedding = self.model_interface.encode(text)
                else:
                    embedding = self.model_interface(text)
                
                embedding = self._normalize(np.array(embedding, dtype=np.float32))
                if use_cache:
                    np.save(self._get_cache_path(text), embedding)
                return embedding
            except Exception as e:
                print(f"⚠️ Interface embedding failed: {e}. Falling back.")

        # Try Sentence-Transformer
        if self.use_local_transformer and self.transformer_model:
            try:
                embedding = self.transformer_model.encode(text, convert_to_numpy=True)
                embedding = self._normalize(embedding.astype(np.float32))
                if use_cache:
                    np.save(self._get_cache_path(text), embedding)
                return embedding
            except Exception as e:
                print(f"⚠️ Transformer error: {e}. Falling back.")

        # Fallback to Johnson-Lindenstrauss
        return self._encode_fallback(text)

    def _encode_fallback(self, text: str) -> np.ndarray:
        features = np.zeros(1024, dtype=np.float32)
        for i, char in enumerate(text):
            idx = ord(char) % 1024
            features[idx] += 1.0
            features[(idx + i) % 1024] += 0.5
        v = np.dot(features, self.projection)
        return self._normalize(v)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12) if norm > 0 else v

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))

# Backward compatibility wrapper
class EmbeddingEngine(AdvancedEmbeddingEngine):
    def __init__(self, dim: int = 384):
        super().__init__(dim=dim)
