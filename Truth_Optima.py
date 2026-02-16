"""
TRUTHOPTIMA - The Complete Hybrid System
===========================================
Combines:
1. Truth API Byzantine Consensus (for high-risk verification)
2. LEO Optima Smart Routing (novelty + coherence)
3. Optima+ Semantic Cache (fast lookup)

Cost-intelligent AI router with Byzantine verification and Verifiable Proof Fragments.
"""

import numpy as np
import hashlib
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

# Import the API interfaces
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
    admm_steps: int = 12         # Increased for better convergence
    A: int = 8                   # Auxiliary states for coherence ADMM
    
    # Cache parameters
    cache_max: int = 2000
    tau_cache: float = 0.82      # Cache similarity threshold
    tau_gap: float = 0.10        # Disambiguation margin
    
    # Trust & verification
    tau_sigma: float = 0.87      # Semantic consistency requirement
    eta: float = 0.85            # Trust evolution rate
    eps: float = 0.05            # LCS tolerance
    
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


@dataclass
class ProofFragment:
    """Verifiable optimization proof as per Section 6 of Whitepaper"""
    lcs: bool             # Local Constraint Satisfaction
    commitment: str       # Commitment Hash H(cost_opt)
    sigma: float          # Semantic Consistency sim(a_opt, a_base)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_valid(self, tau_sigma: float) -> bool:
        return self.lcs and self.sigma >= tau_sigma


# ============================================================
# LIGHTWEIGHT SEMANTIC ENGINE
# ============================================================

class EmbeddingEngine:
    """
    Semantic embedding using a fixed random projection (Johnson-Lindenstrauss).
    This provides stable, semi-semantic embeddings without heavy dependencies.
    """
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        # Initialize a stable projection matrix based on a fixed seed
        state = np.random.RandomState(42)
        self.projection = state.randn(1024, dim).astype(np.float32)
        # Normalize projection for stable output
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)
    
    def encode(self, text: str) -> np.ndarray:
        """Generate a stable embedding using character-level features and random projection"""
        # Create a simple 1024-dim feature vector from text
        features = np.zeros(1024, dtype=np.float32)
        text_bytes = text.encode('utf-8')
        for i, b in enumerate(text_bytes):
            features[i % 1024] += b
            # Add some positional signal
            features[(i * 31) % 1024] += 1.0
        
        # Project to target dimension
        v = np.dot(features, self.projection)
        return self._normalize(v)
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12) if norm > 0 else v
    
    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))


# ============================================================
# MICRO-MEMORY MANAGER
# ============================================================

class MicroMemory:
    """Manages contextual micro-memory with decay (Section 2.3)"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        self.memory: List[np.ndarray] = []
    
    def influence(self, v_raw: np.ndarray) -> np.ndarray:
        """v = v_raw + α * sum(sim(v_raw, m) * m)"""
        if not self.memory:
            return v_raw
        
        mem_stack = np.stack(self.memory)
        # Cosine similarity is just dot product for normalized vectors
        sims = np.dot(mem_stack, v_raw)
        
        influence = np.zeros_like(v_raw)
        for s, m in zip(sims, self.memory):
            if s > 0: # Only positive influence
                influence += s * m
                
        v_enriched = v_raw + self.config.alpha * influence
        return self._normalize(v_enriched)
    
    def update(self, v: np.ndarray):
        """M_k ← β M_k + (1-β)v"""
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
# NOVELTY & COHERENCE ENGINES
# ============================================================

class NoveltyEngine:
    """Measures query novelty via entropy + deviation (Section 3)"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        self.cluster_centers: Optional[np.ndarray] = None
    
    def fit_clusters(self, corpus_embeddings: np.ndarray, n_clusters: int = 8):
        # Using a simple K-means implementation or random centroids for now
        # to avoid sklearn dependency issues
        idx = np.random.choice(len(corpus_embeddings), n_clusters, replace=False)
        self.cluster_centers = corpus_embeddings[idx]
    
    def compute(self, v: np.ndarray) -> float:
        """N(p) = H(p) + λ·D(p)"""
        if self.cluster_centers is None:
            return 0.5
        
        # Dists to cluster centers
        dists = np.linalg.norm(self.cluster_centers - v, axis=1)
        
        # Softmax-like probability for regional entropy
        probs = np.exp(-dists)
        probs /= (np.sum(probs) + 1e-12)
        
        H = -np.sum(probs * np.log(probs + 1e-12))
        D = float(np.min(dists))
        
        return float(H + self.config.lambda_ * D)


class CoherenceEngine:
    """ADMM-based coherence measurement (Section 4)"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
    
    def compute(self, v: np.ndarray) -> float:
        """C(p) = ||v - z|| where z is ADMM consensus"""
        A = self.config.A
        T = self.config.admm_steps
        d = len(v)
        
        # Initialize auxiliary states θ_i and dual variables u_i
        theta = np.stack([v + 0.01 * np.random.randn(d) for _ in range(A)])
        u = np.zeros_like(theta)
        
        # ADMM Iterations
        for _ in range(T):
            # θ_i update (proximal step)
            theta = theta - 0.1 * (theta - v + u)
            # z update (geometric median/mean)
            z = np.mean(theta + u, axis=0)
            # u_i update
            u = u + theta - z
            
        return float(np.linalg.norm(v - z))


# ============================================================
# SEMANTIC CACHE & CONSENSUS
# ============================================================

class SemanticCache:
    """Trust-weighted semantic cache (Section 5)"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        self.data: List[Dict] = [] # (vector, answer, trust)
    
    def lookup(self, v: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.data:
            return None, 0.0
        
        scores = []
        for entry in self.data:
            sim = np.dot(entry['vector'], v)
            score = entry['trust'] * sim
            scores.append((score, entry['answer']))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        s_max, best_ans = scores[0]
        s_second = scores[1][0] if len(scores) > 1 else 0.0
        
        # Cache Acceptance Rule (Section 5.2)
        if s_max >= self.config.tau_cache and (s_max - s_second) >= self.config.tau_gap:
            return best_ans, float(s_max)
        return None, float(s_max)

    def add(self, v: np.ndarray, answer: str, trust: float):
        self.data.append({
            'vector': v,
            'answer': answer,
            'trust': trust,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.data) > self.config.cache_max:
            self.data.pop(0)


class ByzantineConsensus:
    """Byzantine-robust consensus for multi-model verification"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        
    def resolve(self, embeddings: List[np.ndarray], trust_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """Compute consensus vector and update trust weights"""
        n = len(embeddings)
        stack = np.stack(embeddings)
        
        # Weighted mean as baseline consensus
        consensus = np.average(stack, axis=0, weights=trust_weights)
        
        # Detect outliers
        dists = np.linalg.norm(stack - consensus, axis=1)
        median_dist = np.median(dists)
        outliers = dists > (self.config.outlier_threshold * median_dist + 1e-6)
        
        # Update trust (Section 8)
        new_trust = trust_weights.copy()
        for i in range(n):
            if outliers[i]:
                new_trust[i] *= self.config.trust_decay
            else:
                # Reward consistency
                sim = np.dot(embeddings[i], consensus)
                new_trust[i] = self.config.eta * new_trust[i] + (1 - self.config.eta) * sim
                
        return consensus, new_trust, outliers.tolist()


# ============================================================
# RISK ASSESSMENT
# ============================================================

class RiskAssessor:
    """Analyzes prompt risk level based on keywords"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
    
    def assess(self, prompt: str) -> RiskLevel:
        prompt_lower = prompt.lower()
        
        hits = 0
        for category, keywords in self.config.high_risk_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                hits += 1
        
        if hits >= 2: return RiskLevel.CRITICAL
        if hits == 1: return RiskLevel.HIGH
        return RiskLevel.LOW


# ============================================================
# MAIN TRUTHOPTIMA ENGINE
# ============================================================

@dataclass
class TruthOptimaResponse:
    answer: str
    route: RouteType
    risk_level: RiskLevel
    confidence: float
    cost_estimate: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    novelty: Optional[float] = None
    coherence: Optional[float] = None
    cache_score: Optional[float] = None
    sigma: Optional[float] = None
    commit: Optional[str] = None
    proof: Optional[ProofFragment] = None
    model_responses: Optional[Dict[str, str]] = None
    outliers: List[str] = field(default_factory=list)
    trust_scores: Dict[str, float] = field(default_factory=dict)


class TruthOptima:
    """The LEO Optima Core Engine"""
    
    def __init__(self, config: Optional[TruthOptimaConfig] = None, models: Optional[Dict[str, LLMInterface]] = None):
        self.config = config or TruthOptimaConfig()
        self.embedder = EmbeddingEngine(dim=self.config.dim)
        self.memory = MicroMemory(self.config)
        self.novelty = NoveltyEngine(self.config)
        self.coherence = CoherenceEngine(self.config)
        self.cache = SemanticCache(self.config)
        self.consensus = ByzantineConsensus(self.config)
        self.risk_assessor = RiskAssessor(self.config)
        
        # Default models if none provided
        self.models = models or {
            'gpt4_sim': LLMSimulator("GPT-4"),
            'claude_sim': LLMSimulator("Claude-3"),
            'llama_sim': LLMSimulator("Llama-3")
        }
        
        # Initial trust weights
        self.trust_weights = {name: 0.9 for name in self.models.keys()}
        
        # Persistent Stats
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'fast_routes': 0,
            'consensus_routes': 0,
            'total_cost': 0.0
        }

    def initialize_novelty(self, corpus: List[str]):
        embs = np.stack([self.embedder.encode(t) for t in corpus])
        self.novelty.fit_clusters(embs)

    async def ask(self, question: str) -> TruthOptimaResponse:
        self.stats['total_queries'] += 1
        
        # 1. Embedding + Memory Influence (Section 2)
        v_raw = self.embedder.encode(question)
        v = self.memory.influence(v_raw)
        
        # 2. Cache Lookup (Section 5)
        cached_ans, cache_score = self.cache.lookup(v)
        if cached_ans:
            self.stats['cache_hits'] += 1
            return TruthOptimaResponse(
                answer=cached_ans,
                route=RouteType.CACHE,
                risk_level=self.risk_assessor.assess(question),
                confidence=1.0,
                cost_estimate=0.0,
                cache_score=cache_score
            )
        
        # 3. Novelty & Coherence (Section 3 & 4)
        N = self.novelty.compute(v)
        C = self.coherence.compute(v)
        risk = self.risk_assessor.assess(question)
        
        # 4. Routing Policy (Section 7)
        use_consensus = (risk != RiskLevel.LOW) or (N >= self.config.gamma) or (C >= self.config.delta)
        
        if use_consensus:
            resp = await self._consensus_route(question, v, N, C, risk)
            self.stats['consensus_routes'] += 1
        else:
            resp = await self._fast_route(question, v, N, C, risk)
            self.stats['fast_routes'] += 1
            
        self.stats['total_cost'] += resp.cost_estimate
        # Update memory and cache
        self.memory.update(v)
        return resp

    async def _fast_route(self, question: str, v: np.ndarray, N: float, C: float, risk: RiskLevel) -> TruthOptimaResponse:
        # Pick most trusted model
        best_model_name = max(self.trust_weights.items(), key=lambda x: x[1])[0]
        model = self.models[best_model_name]
        
        answer = await model.query(question)
        
        # Generate Proof Fragment (Section 6)
        # LCS simulated: cost difference within tolerance
        lcs = True 
        commit = hashlib.sha256(answer.encode()).hexdigest()[:16]
        v_ans = self.embedder.encode(answer)
        sigma = self.embedder.similarity(v, v_ans)
        
        proof = ProofFragment(lcs=lcs, commitment=commit, sigma=sigma)
        
        if proof.is_valid(self.config.tau_sigma):
            self.cache.add(v, answer, self.trust_weights[best_model_name])
            
        return TruthOptimaResponse(
            answer=answer,
            route=RouteType.FAST,
            risk_level=risk,
            confidence=sigma,
            cost_estimate=0.0001,
            novelty=N,
            coherence=C,
            sigma=sigma,
            commit=commit,
            proof=proof
        )

    async def _consensus_route(self, question: str, v: np.ndarray, N: float, C: float, risk: RiskLevel) -> TruthOptimaResponse:
        tasks = [m.query(question) for m in self.models.values()]
        answers = await asyncio.gather(*tasks)
        model_names = list(self.models.keys())
        
        embs = [self.embedder.encode(a) for a in answers]
        weights = np.array([self.trust_weights[name] for name in model_names])
        
        consensus_vec, new_weights, outliers = self.consensus.resolve(embs, weights)
        
        # Update system trust
        for i, name in enumerate(model_names):
            self.trust_weights[name] = float(new_weights[i])
            
        # Select best answer (closest to consensus)
        sims = [np.dot(e, consensus_vec) for e in embs]
        best_idx = np.argmax(sims)
        answer = answers[best_idx]
        
        # Proof Fragment
        commit = hashlib.sha256(answer.encode()).hexdigest()[:16]
        sigma = float(np.dot(v, self.embedder.encode(answer)))
        proof = ProofFragment(lcs=True, commitment=commit, sigma=sigma)
        
        if proof.is_valid(self.config.tau_sigma):
            self.cache.add(v, answer, 1.0)
            
        return TruthOptimaResponse(
            answer=answer,
            route=RouteType.CONSENSUS,
            risk_level=risk,
            confidence=float(np.mean(sims)),
            cost_estimate=0.005,
            novelty=N,
            coherence=C,
            sigma=sigma,
            commit=commit,
            proof=proof,
            model_responses=dict(zip(model_names, answers)),
            outliers=[model_names[i] for i, is_out in enumerate(outliers) if is_out],
            trust_scores=self.trust_weights.copy()
        )

# ============================================================
# DEMO
# ============================================================

async def demo():
    system = TruthOptima()
    system.initialize_novelty(["tech", "medicine", "law", "finance", "science", "art"])
    
    queries = [
        "What is quantum entanglement?",
        "Is it safe to take ibuprofen with aspirin?", # Risk
        "What is quantum entanglement?", # Cache hit
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        resp = await system.ask(q)
        print(f"Route: {resp.route.value} | Confidence: {resp.confidence:.3f}")
        if resp.proof:
            print(f"Proof Valid: {resp.proof.is_valid(system.config.tau_sigma)}")

if __name__ == "__main__":
    asyncio.run(demo())
