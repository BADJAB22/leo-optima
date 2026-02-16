import numpy as np
import hashlib
import json
import os
from typing import Optional, Tuple, Dict
from datetime import datetime

class SimpleEmbedding:
    """Deterministic embedding for local processing without external dependencies"""
    def __init__(self, dim: int = 384):
        self.dim = dim
    
    def encode(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        v = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))

class PersistentCache:
    """A file-based semantic cache that survives restarts"""
    def __init__(self, cache_file: str = "leo_cache.json", threshold: float = 0.92):
        self.cache_file = cache_file
        self.threshold = threshold
        self.embedder = SimpleEmbedding()
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.data, f)

    def lookup(self, query: str) -> Tuple[Optional[str], float]:
        if not self.data:
            return None, 0.0
        
        v_query = self.embedder.encode(query)
        best_match = None
        max_sim = 0.0

        for entry in self.data:
            v_entry = np.array(entry['vector'])
            sim = self.embedder.similarity(v_query, v_entry)
            if sim > max_sim:
                max_sim = sim
                best_match = entry['answer']

        if max_sim >= self.threshold:
            return best_match, max_sim
        return None, max_sim

    def add(self, query: str, answer: str):
        v_query = self.embedder.encode(query)
        self.data.append({
            'query': query,
            'answer': answer,
            'vector': v_query.tolist(),
            'timestamp': datetime.now().isoformat()
        })
        self._save()

class LeoOptimizer:
    """The core logic that decides if we need to call the expensive model"""
    def __init__(self, cache_threshold: float = 0.92):
        self.cache = PersistentCache(threshold=cache_threshold)
        
    def process_query(self, query: str) -> Tuple[Optional[str], Dict]:
        # Check cache first
        cached_answer, score = self.cache.lookup(query)
        
        metrics = {
            "cache_score": round(score, 4),
            "optimized": cached_answer is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        return cached_answer, metrics

    def update_cache(self, query: str, answer: str):
        self.cache.add(query, answer)
