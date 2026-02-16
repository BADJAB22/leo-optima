"""
TRUTHOPTIMA - The Complete Hybrid System
===========================================
Combines:
1. Truth API Byzantine Consensus (for high-risk verification)
2. LEO Optima Smart Routing (novelty + coherence)
3. Optima+ HNSW Cache (fast semantic lookup)

Cost-intelligent AI router with Byzantine verification when needed.
"""

import numpy as np
import hashlib
import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import hnswlib
except ImportError:
    # Mocking for environments without these heavy dependencies
    class KMeans:
        def __init__(self, n_clusters=8, random_state=42): self.cluster_centers_ = None
        def fit(self, X): self.cluster_centers_ = np.mean(X, axis=0, keepdims=True)
    
    def cosine_similarity(X, Y):
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        return np.dot(X_norm, Y_norm.T)

    class MockIndex:
        def __init__(self, space, dim): self.dim = dim; self.data = []
        def init_index(self, **kwargs): pass
        def set_ef(self, ef): pass
        def add_items(self, v, labels): self.data.append((v, labels[0]))
        def knn_query(self, v, k):
            if not self.data: return [], []
            sims = [np.dot(v[0], d[0][0]) for d in self.data]
            idx = np.argmax(sims)
            return [[self.data[idx][1]]], [[1.0 - sims[idx]]]
        def mark_deleted(self, idx): pass

    class hnswlib:
        Index = MockIndex
import uuid
import os

# Import the new API interfaces
from api_interfaces import LLMInterface, LLMSimulator, OpenAICompatibleAPI

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class TruthOptimaConfig:
    """Unified system configuration"""
    # Embedding
    dim: int = 384
    
    # Memory parameters
    alpha: float = 0.12          # Memory influence weight
    beta: float = 0.96           # Memory decay factor
    memory_max: int = 32
    
    # Novelty parameters
    lambda_: float = 0.28        # Deviation weighting
    gamma: float = 0.45          # Novelty threshold (SMALL route if below)
    
    # Coherence parameters
    delta: float = 0.55          # Coherence threshold (SMALL route if below)
    admm_steps: int = 8
    A: int = 6                   # Auxiliary states for coherence ADMM
    
    # Cache parameters
    cache_max: int = 2000
    tau_cache: float = 0.82      # Cache similarity threshold
    tau_gap: float = 0.10        # Disambiguation margin
    
    # Trust & verification
    tau_sigma: float = 0.87      # Semantic consistency requirement
    eta: float = 0.85            # Trust evolution rate
    
    # Byzantine consensus (for CONSENSUS route)
    rho: float = 0.5             # ADMM penalty
    consensus_iterations: int = 10
    outlier_threshold: float = 2.0
    trust_decay: float = 0.95
    
    # Risk assessment
    high_risk_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'medical': ['medical', 'health', 'disease', 'drug', 'medicine', 'treatment', 'pregnancy', 'aspirin'],
        'legal': ['legal', 'law', 'court', 'sue', 'contract', 'rights'],
        'financial': ['invest', 'stock', 'crypto', 'loan', 'tax', 'money'],
        'safety': ['safe', 'danger', 'risk', 'emergency', 'hazard']
    })


class RouteType(Enum):
    """Routing decisions"""
    CACHE = "CACHE"           # Cache hit (free!)
    FAST = "FAST"             # Single small model (cheap)
    CONSENSUS = "CONSENSUS"   # Byzantine verification (expensive but verified)


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================
# EMBEDDING ENGINE
# ============================================================

class EmbeddingEngine:
    """Semantic embedding with deterministic simulation"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
    
    def encode(self, text: str) -> np.ndarray:
        """Generate deterministic embedding"""
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        v = np.random.randn(self.dim).astype(np.float32)
        
        # Add semantic signals
        text_lower = text.lower()
        if "not" in text_lower or "avoid" in text_lower:
            v[:10] -= 0.5
        if "recommend" in text_lower or "safe" in text_lower:
            v[:10] += 0.5
        if "consult" in text_lower or "doctor" in text_lower:
            v[10:20] += 0.3
            
        return self._normalize(v)
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12) if norm > 0 else v
    
    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))


# ============================================================
# MICRO-MEMORY MANAGER
# ============================================================

class MicroMemory:
    """Manages contextual micro-memory with decay"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        self.memory: List[np.ndarray] = []
    
    def influence(self, v_raw: np.ndarray) -> np.ndarray:
        """Apply memory influence to embedding"""
        if not self.memory:
            return v_raw
        
        mem_stack = np.stack(self.memory)
        sims = cosine_similarity([v_raw], mem_stack)[0]
        influence = np.sum([s * m for s, m in zip(sims, self.memory)], axis=0)
        v_enriched = v_raw + self.config.alpha * influence
        
        return self._normalize(v_enriched)
    
    def update(self, v: np.ndarray):
        """Decay update to memory"""
        if len(self.memory) < self.config.memory_max:
            self.memory.append(v)
        else:
            # Decay all memories toward new vector
            self.memory = [
                self._normalize(self.config.beta * m + (1 - self.config.beta) * v)
                for m in self.memory
            ]
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12) if norm > 0 else v


# ============================================================
# NOVELTY ENGINE
# ============================================================

class NoveltyEngine:
    """Measures query novelty via entropy + deviation"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        self.cluster_centers: Optional[np.ndarray] = None
    
    def fit_clusters(self, corpus_embeddings: np.ndarray, n_clusters: int = 8):
        """Fit K-means clusters for novelty measurement"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(corpus_embeddings)
        self.cluster_centers = kmeans.cluster_centers_.astype(np.float32)
    
    def compute(self, v: np.ndarray) -> float:
        """N(p) = H(p) + λ·D(p)"""
        if self.cluster_centers is None:
            return 0.5  # Neutral
        
        # Compute distances to clusters
        dists = np.linalg.norm(self.cluster_centers - v, axis=1)
        
        # Soft cluster assignment
        weights = np.exp(-dists)
        q = weights / (np.sum(weights) + 1e-12)
        
        # Regional entropy
        H = -np.sum(q * np.log(q + 1e-12))
        
        # Minimum deviation
        D = float(np.min(dists))
        
        return float(H + self.config.lambda_ * D)


# ============================================================
# COHERENCE ENGINE
# ============================================================

class CoherenceEngine:
    """ADMM-based coherence measurement"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
    
    def compute(self, v: np.ndarray) -> float:
        """C(p) = ||v - z|| where z is ADMM consensus"""
        A = self.config.A
        T = self.config.admm_steps
        d = len(v)
        
        # Initialize auxiliary states
        theta = np.stack([
            v + 0.03 * np.random.randn(d).astype(np.float32)
            for _ in range(A)
        ])
        u = np.zeros_like(theta)
        
        # ADMM iterations
        for _ in range(T):
            theta = theta - 0.08 * (theta - v)
            z = np.median(theta + u, axis=0)
            u = u + theta - z
        
        return float(np.linalg.norm(v - z))


# ============================================================
# HNSW SEMANTIC CACHE
# ============================================================

@dataclass
class CacheEntry:
    """Cache entry with trust score"""
    answer: str
    trust: float
    timestamp: str


class SemanticCache:
    """Fast HNSW-based semantic cache with trust weighting"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        self.dim = config.dim
        
        self.entries: List[CacheEntry] = []
        self.vectors: List[np.ndarray] = []
        self.lru: List[int] = []
        self.count = 0
        
        # HNSW index
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(
            max_elements=config.cache_max,
            ef_construction=200,
            M=48
        )
        self.index.set_ef(128)
    
    def lookup(self, v: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Cache lookup with disambiguation"""
        if self.count == 0:
            return None, None
        
        k = min(5, self.count)
        query = np.asarray([v], dtype=np.float32)
        labels, dists = self.index.knn_query(query, k=k)
        labels = labels[0]
        sims = 1.0 - dists[0]  # cosine similarity
        
        # Trust-weighted scores
        trust_scores = [self.entries[i].trust * sims[j] for j, i in enumerate(labels)]
        
        best_idx = int(np.argmax(trust_scores))
        best_score = float(trust_scores[best_idx])
        second = float(np.partition(trust_scores, -2)[-2]) if len(trust_scores) > 1 else -1.0
        
        # Check acceptance conditions
        if best_score >= self.config.tau_cache and (best_score - second) >= self.config.tau_gap:
            idx = labels[best_idx]
            self.lru[idx] = 0  # Reset LRU
            return self.entries[idx].answer, best_score
        
        return None, best_score
    
    def add(self, v: np.ndarray, answer: str, trust: float):
        """Add entry with LRU eviction"""
        if trust < 0.4:
            return
        
        v = np.asarray(v, dtype=np.float32)
        entry = CacheEntry(
            answer=answer,
            trust=trust,
            timestamp=datetime.now().isoformat()
        )
        
        if self.count < self.config.cache_max:
            idx = self.count
            self.entries.append(entry)
            self.vectors.append(v)
            self.lru.append(0)
            self.index.add_items(v.reshape(1, -1), [idx])
            self.count += 1
        else:
            # LRU eviction
            evict_idx = int(np.argmax(self.lru))
            self.entries[evict_idx] = entry
            self.vectors[evict_idx] = v
            
            try:
                self.index.mark_deleted(evict_idx)
            except:
                pass
            
            self.index.add_items(v.reshape(1, -1), [evict_idx])
            self.lru[evict_idx] = 0
        
        # Age all entries
        self.lru = [x + 1 for x in self.lru]


# ============================================================
# BYZANTINE CONSENSUS ENGINE
# ============================================================

class ByzantineConsensus:
    """Byzantine-resilient ADMM consensus for verification"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
    
    def geometric_median(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Weighted geometric median (outlier-resistant)"""
        median = np.average(points, axis=0, weights=weights)
        
        for _ in range(20):
            dists = np.linalg.norm(points - median, axis=1)
            dists = np.maximum(dists, 1e-8)
            w = weights / dists
            w = w / (np.sum(w) + 1e-12)
            median = np.sum(points * w[:, np.newaxis], axis=0)
        
        return median
    
    def consensus(
        self,
        embeddings: List[np.ndarray],
        trust_weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """Run Byzantine consensus"""
        X = np.stack(embeddings)
        
        theta = X.copy()
        u = np.zeros_like(X)
        w = self.geometric_median(X, trust_weights)
        
        for _ in range(self.config.consensus_iterations):
            theta = (self.config.rho * w + u) / (1 + self.config.rho)
            w = self.geometric_median(theta - u, trust_weights)
            u = u + theta - w
            
            if np.linalg.norm(w - theta[0]) < 1e-4:
                break
        
        # Outlier detection
        distances = np.linalg.norm(X - w, axis=1)
        mean_dist, std_dist = np.mean(distances), np.std(distances)
        outliers = distances > (mean_dist + self.config.outlier_threshold * std_dist)
        
        # Update trust
        trust_weights = trust_weights.copy()
        trust_weights[outliers] *= self.config.trust_decay
        trust_weights /= (np.sum(trust_weights) + 1e-12)
        
        return w, trust_weights, outliers.tolist()


# ============================================================
# RISK ASSESSOR
# ============================================================

class RiskAssessor:
    """Analyzes query risk based on keywords"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
    
    def assess(self, question: str) -> RiskLevel:
        q_lower = question.lower()
        
        # Critical keywords
        critical = ['suicide', 'kill', 'bomb', 'attack', 'illegal']
        if any(kw in q_lower for kw in critical):
            return RiskLevel.CRITICAL
        
        # High-risk domains
        for domain, keywords in self.config.high_risk_keywords.items():
            if any(kw in q_lower for kw in keywords):
                return RiskLevel.HIGH
        
        return RiskLevel.LOW


# ============================================================
# MAIN TRUTHOPTIMA SYSTEM
# ============================================================

@dataclass
class TruthOptimaResponse:
    """Complete system response"""
    answer: str
    route: RouteType
    risk_level: RiskLevel
    confidence: float
    cost_estimate: float
    
    # Metrics
    novelty: Optional[float] = None
    coherence: Optional[float] = None
    cache_score: Optional[float] = None
    
    # Verification (for CONSENSUS route)
    model_responses: Optional[Dict[str, str]] = None
    outliers: Optional[List[str]] = None
    trust_scores: Optional[Dict[str, float]] = None
    
    # Proof
    sigma: Optional[float] = None
    commit: Optional[str] = None
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TruthOptima:
    """
    Complete TruthOptima System
    
    Smart routing with Byzantine verification for high-risk queries.
    """
    
    def __init__(self, config: Optional[TruthOptimaConfig] = None, models: Optional[Dict[str, LLMInterface]] = None):
        self.config = config or TruthOptimaConfig()
        
        # Core components
        self.embedder = EmbeddingEngine(self.config.dim)
        self.memory = MicroMemory(self.config)
        self.novelty = NoveltyEngine(self.config)
        self.coherence = CoherenceEngine(self.config)
        self.cache = SemanticCache(self.config)
        self.consensus = ByzantineConsensus(self.config)
        self.risk_assessor = RiskAssessor(self.config)
        
        # LLM models (passed from outside or use simulators by default)
        self.models = models or {
            'gpt4': LLMSimulator('GPT-4'),
            'claude': LLMSimulator('Claude'),
            'gemini': LLMSimulator('Gemini'),
            'llama': LLMSimulator('Llama'),
        }
        
        # Trust tracking
        self.trust_weights = {
            name: 1.0 / len(self.models)
            for name in self.models.keys()
        }
        
        # Stats & Analytics
        self.stats_file = "leo_analytics.json"
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict:
        """Load persistent statistics from disk"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'total_queries': 0,
            'cache_hits': 0,
            'fast_routes': 0,
            'consensus_routes': 0,
            'total_cost_saved': 0.0,
            'total_cost_spent': 0.0,
            'avg_latency_saved_ms': 0.0,
            'history': [] # Recent events for dashboard
        }

    def _save_stats(self):
        """Persist statistics to disk"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def _update_metrics(self, route: RouteType, cost_spent: float, cost_saved: float, latency_ms: float = 0):
        """Update internal metrics and persist"""
        self.stats['total_queries'] += 1
        if route == RouteType.CACHE:
            self.stats['cache_hits'] += 1
        elif route == RouteType.FAST:
            self.stats['fast_routes'] += 1
        elif route == RouteType.CONSENSUS:
            self.stats['consensus_routes'] += 1
            
        self.stats['total_cost_spent'] += cost_spent
        self.stats['total_cost_saved'] += cost_saved
        
        # Keep last 50 events for history
        event = {
            "timestamp": datetime.now().isoformat(),
            "route": route.value,
            "cost_spent": cost_spent,
            "cost_saved": cost_saved
        }
        self.stats['history'] = [event] + self.stats['history'][:49]
        self._save_stats()
    
    def initialize_novelty(self, corpus: List[str]):
        """Initialize novelty clusters with sample corpus"""
        embeddings = np.stack([self.embedder.encode(text) for text in corpus])
        self.novelty.fit_clusters(embeddings, n_clusters=8)
    
    async def ask(self, question: str) -> TruthOptimaResponse:
        """Main routing function with analytics tracking"""
        # ===== STEP 1: Embedding + Memory =====
        v_raw = self.embedder.encode(question)
        v = self.memory.influence(v_raw)
        
        # ===== STEP 2: Cache Lookup =====
        cached_answer, cache_score = self.cache.lookup(v)
        if cached_answer is not None:
            self.memory.update(v)
            # Estimated cost saved: average cost of a standard LLM call (~$0.01)
            self._update_metrics(RouteType.CACHE, cost_spent=0.0, cost_saved=0.01)
            
            return TruthOptimaResponse(
                answer=cached_answer,
                route=RouteType.CACHE,
                risk_level=self.risk_assessor.assess(question),
                confidence=1.0,
                cost_estimate=0.0,
                cache_score=cache_score
            )
        
        # ===== STEP 3: Compute Metrics =====
        N = self.novelty.compute(v)
        C = self.coherence.compute(v)
        risk = self.risk_assessor.assess(question)
        
        # ===== STEP 4: Routing Decision =====
        use_consensus = (
            risk in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            N >= self.config.gamma or
            C >= self.config.delta
        )
        
        if use_consensus:
            result = await self._consensus_route(question, v, N, C, risk)
            # Consensus is expensive, but we save compared to manual verification
            self._update_metrics(RouteType.CONSENSUS, cost_spent=result.cost_estimate, cost_saved=0.05)
        else:
            result = await self._fast_route(question, v, N, C, risk)
            # Fast route saves by using a smaller model
            self._update_metrics(RouteType.FAST, cost_spent=result.cost_estimate, cost_saved=0.009)
        
        # ===== STEP 5: Update System State =====
        self.memory.update(v)
        
        return result
    
    async def _fast_route(self, question: str, v: np.ndarray, N: float, C: float, risk: RiskLevel) -> TruthOptimaResponse:
        """Fast route: single best model"""
        best_model_name = max(self.trust_weights.items(), key=lambda x: x[1])[0]
        best_model = self.models[best_model_name]
        
        answer = await best_model.query(question)
        
        v_answer = self.embedder.encode(answer)
        sigma = self.embedder.similarity(v, v_answer)
        commit = hashlib.sha256(answer.encode()).hexdigest()[:16]
        
        trust = float(np.clip(0.5 + 0.4 * sigma, 0, 1))
        if sigma >= self.config.tau_sigma and trust >= 0.4:
            self.cache.add(v, answer, trust)
        
        return TruthOptimaResponse(
            answer=answer,
            route=RouteType.FAST,
            risk_level=risk,
            confidence=float(sigma),
            cost_estimate=0.0001,
            novelty=N,
            coherence=C,
            sigma=sigma,
            commit=commit
        )
    
    async def _consensus_route(self, question: str, v: np.ndarray, N: float, C: float, risk: RiskLevel) -> TruthOptimaResponse:
        """Consensus route: Byzantine verification"""
        responses = await asyncio.gather(*[
            model.query(question)
            for model in self.models.values()
        ])
        
        model_responses = dict(zip(self.models.keys(), responses))
        embeddings = [self.embedder.encode(resp) for resp in responses]
        
        trust_array = np.array([self.trust_weights[name] for name in self.models.keys()])
        consensus_vec, updated_trust, outliers = self.consensus.consensus(embeddings, trust_array)
        
        for i, name in enumerate(self.models.keys()):
            self.trust_weights[name] = float(updated_trust[i])
        
        similarities = [self.embedder.similarity(consensus_vec, emb) for emb in embeddings]
        best_idx = int(np.argmax(similarities))
        answer = responses[best_idx]
        
        distances = [np.linalg.norm(emb - consensus_vec) for emb in embeddings]
        confidence = 1.0 / (1.0 + np.std(distances))
        
        v_answer = self.embedder.encode(answer)
        sigma = self.embedder.similarity(v, v_answer)
        commit = hashlib.sha256(answer.encode()).hexdigest()[:16]
        
        outlier_names = [name for name, is_out in zip(self.models.keys(), outliers) if is_out]
        
        trust = float(np.clip(0.5 + 0.4 * sigma, 0, 1))
        if sigma >= self.config.tau_sigma and confidence >= 0.7:
            self.cache.add(v, answer, trust)
        
        return TruthOptimaResponse(
            answer=answer,
            route=RouteType.CONSENSUS,
            risk_level=risk,
            confidence=float(confidence),
            cost_estimate=0.005,
            novelty=N,
            coherence=C,
            model_responses=model_responses,
            outliers=outlier_names,
            trust_scores=self.trust_weights.copy(),
            sigma=sigma,
            commit=commit
        )
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        total = self.stats['total_queries']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'cache_hit_rate': self.stats['cache_hits'] / total,
            'fast_route_rate': self.stats['fast_routes'] / total,
            'consensus_route_rate': self.stats['consensus_routes'] / total,
            'cache_size': self.cache.count,
            'memory_size': len(self.memory.memory)
        }
    
    def format_response(self, response: TruthOptimaResponse) -> str:
        """Format response as a string"""
        lines = []
        lines.append("\n" + "="*70)
        lines.append(f"🎯 TRUTHOPTIMA RESPONSE")
        lines.append("="*70)
        lines.append(f"\n✅ ANSWER: {response.answer}")
        lines.append(f"\n📍 ROUTE: {response.route.value}")
        lines.append(f"⚠️  RISK: {response.risk_level.value}")
        lines.append(f"📊 CONFIDENCE: {response.confidence:.1%}")
        lines.append(f"💰 COST: ${response.cost_estimate:.4f}")
        
        if response.novelty is not None:
            lines.append(f"\n📈 METRICS:")
            lines.append(f"   Novelty: {response.novelty:.3f}")
            lines.append(f"   Coherence: {response.coherence:.3f}")
        
        if response.cache_score is not None:
            lines.append(f"   Cache Score: {response.cache_score:.3f}")
        
        if response.sigma is not None:
            lines.append(f"\n🔐 VERIFICATION:")
            lines.append(f"   Sigma: {response.sigma:.3f}")
            lines.append(f"   Commit: {response.commit}")
        
        if response.outliers:
            lines.append(f"\n⚠️  OUTLIERS: {', '.join(response.outliers)}")
        
        if response.trust_scores:
            lines.append(f"\n🔐 TRUST SCORES:")
            for model, trust in sorted(response.trust_scores.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(trust * 20)
                lines.append(f"   {model:8s} {bar} {trust:.3f}")
        
        lines.append("\n" + "="*70)
        return "\n".join(lines)

    def print_response(self, response: TruthOptimaResponse):
        """Pretty print response"""
        print(self.format_response(response))

    def export_response(self, response: TruthOptimaResponse, filename: str):
        """Export response to a text file"""
        content = self.format_response(response)
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Report exported to: {filename}")


# ============================================================
# DEMO
# ============================================================

async def demo():
    """Comprehensive demo"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║                    TRUTHOPTIMA v1.1                      ║
    ║        Universal API Support & Byzantine Consensus       ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Example of how to use real APIs (commented out for demo)
    # models = {
    #     'gpt4': OpenAICompatibleAPI(model_name="gpt-4"),
    #     'claude': OpenAICompatibleAPI(model_name="claude-3-opus", base_url="https://api.anthropic.com/v1"),
    # }
    # system = TruthOptima(models=models)
    
    # For demo, we use simulators
    system = TruthOptima()
    
    # Initialize novelty clusters
    print("\n[1] Initializing novelty clusters...")
    corpus = [
        "machine learning algorithms",
        "quantum computing basics",
        "react component development",
        "database optimization",
        "financial forecasting",
        "medical diagnosis procedures",
        "legal contract analysis",
        "cybersecurity best practices"
    ]
    system.initialize_novelty(corpus)
    print("✓ Clusters initialized")
    
    # Test queries
    queries = [
        "What is the capital of France?",
        "How do I sort an array in Python?",
        "What is the capital of France?",  # Duplicate - should hit cache
        "Is aspirin safe during pregnancy?",  # High-risk medical
        "Can I sue for breach of contract?",  # High-risk legal
        "Explain gradient descent in ML",  # Complex technical
        "What's 2+2?",  # Simple
        "Should I invest in cryptocurrency?",  # High-risk financial
    ]
    
    print("\n[2] Processing queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(queries)}: {query}")
        print('='*70)
        
        response = await system.ask(query)
        system.print_response(response)
        
        await asyncio.sleep(0.5)
    
    # Final statistics
    print("\n" + "="*70)
    print("[3] System Statistics")
    print("="*70)
    stats = system.get_stats()
    
    print(f"\nTotal Queries: {stats['total_queries']}")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
    print(f"FAST Route: {stats['fast_route_rate']:.1%}")
    print(f"CONSENSUS Route: {stats['consensus_route_rate']:.1%}")
    print(f"Cache Size: {stats['cache_size']}")
    print(f"Memory Size: {stats['memory_size']}")
    
    # Cost analysis
    total_cost = (
        stats['cache_hits'] * 0.0 +
        stats['fast_routes'] * 0.0001 +
        stats['consensus_routes'] * 0.005
    )
    
    naive_cost = stats['total_queries'] * 0.005
    savings = (naive_cost - total_cost) / naive_cost * 100
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"   Total Cost: ${total_cost:.4f}")
    print(f"   Naive Cost (all consensus): ${naive_cost:.4f}")
    print(f"   Savings: {savings:.1f}%")
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo())
