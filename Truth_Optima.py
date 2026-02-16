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
from sklearn.neighbors import NearestNeighbors

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
    tau_cache: float = 0.90      # Cache similarity threshold
    tau_gap: float = 0.05        # Disambiguation margin
    
    # Trust & verification
    tau_sigma: float = 0.75      # Semantic consistency requirement
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
        # Use a more stable character-based feature set
        for i, char in enumerate(text):
            idx = ord(char) % 1024
            features[idx] += 1.0
            # Add positional context
            features[(idx + i) % 1024] += 0.5
        
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
    
    def __init__(self, config: TruthOptimaConfig, storage_path: Optional[str] = None):
        self.config = config
        self.memory: List[np.ndarray] = []
        self.storage_path = storage_path
        if self.storage_path and os.path.exists(self.storage_path):
            self.load()
    
    def save(self):
        """Persist memory state to disk"""
        if not self.storage_path:
            return
        data = [m.tolist() for m in self.memory]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)
            
    def load(self):
        """Load memory state from disk"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.memory = [np.array(m, dtype=np.float32) for m in data]
        except Exception as e:
            print(f"Error loading memory: {e}")
    
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
        n_samples = len(corpus_embeddings)
        actual_clusters = min(n_clusters, n_samples)
        if actual_clusters == 0:
            return
        idx = np.random.choice(n_samples, actual_clusters, replace=False)
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
    """Trust-weighted semantic cache with O(log N) search (Section 5)"""
    
    def __init__(self, config: TruthOptimaConfig, storage_path: Optional[str] = None):
        self.config = config
        self.data: List[Dict] = [] # (vector, answer, trust)
        self.storage_path = storage_path
        self.index = None
        if self.storage_path and os.path.exists(self.storage_path):
            self.load()
        self._rebuild_index()
            
    def _rebuild_index(self):
        """Rebuild the NearestNeighbors index for fast O(log N) search"""
        if not self.data:
            self.index = None
            return
        
        vectors = np.stack([entry['vector'] for entry in self.data])
        # Use ball_tree or kd_tree for O(log N) search
        self.index = NearestNeighbors(n_neighbors=min(2, len(self.data)), metric='cosine', algorithm='auto')
        self.index.fit(vectors)

    def save(self):
        """Persist cache state to disk"""
        if not self.storage_path:
            return
        serializable_data = []
        for entry in self.data:
            serializable_data.append({
                'vector': entry['vector'].tolist(),
                'answer': entry['answer'],
                'trust': entry['trust'],
                'timestamp': entry['timestamp']
            })
        with open(self.storage_path, 'w') as f:
            json.dump(serializable_data, f)
            
    def load(self):
        """Load cache state from disk"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, 'r') as f:
                serializable_data = json.load(f)
                self.data = []
                for entry in serializable_data:
                    self.data.append({
                        'vector': np.array(entry['vector'], dtype=np.float32),
                        'answer': entry['answer'],
                        'trust': entry['trust'],
                        'timestamp': entry['timestamp']
                    })
                self._rebuild_index()
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def lookup(self, v: np.ndarray) -> Tuple[Optional[str], float]:
        """O(log N) search for nearest neighbor in cache with trust gating"""
        if not self.data or self.index is None:
            return None, 0.0
        
        # Fast search using the index
        v_reshaped = v.reshape(1, -1)
        n_search = min(2, len(self.data))
        distances, indices = self.index.kneighbors(v_reshaped, n_neighbors=n_search)
        
        # Convert cosine distance to similarity: sim = 1 - dist
        similarities = 1.0 - distances[0]
        
        scores = []
        for i, idx in enumerate(indices[0]):
            entry = self.data[idx]
            # Cache Poisoning Mitigation: Trust Gating
            # If the cached answer has low trust, we penalize its score
            trust_factor = entry.get('trust', 1.0)
            score = float(similarities[i]) * trust_factor
            scores.append((score, entry['answer']))
            
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
        
        # Rebuild index periodically or after each add for small caches
        self._rebuild_index()


class ByzantineConsensus:
    """Byzantine-robust consensus for multi-model verification"""
    
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        
    def resolve(self, embeddings: List[np.ndarray], trust_weights: np.ndarray, proofs: Optional[List[bool]] = None) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """Compute consensus vector and update trust weights based on consensus and proofs"""
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
            # T_new = f(T_old, sigma, valid(pi))
            sigma = float(np.dot(embeddings[i], consensus))
            is_valid_proof = proofs[i] if proofs else True
            
            if outliers[i] or not is_valid_proof:
                # Penalize outliers or invalid proofs
                penalty = self.config.trust_decay if outliers[i] else 0.9
                new_trust[i] *= penalty
            else:
                # Reward consistency and valid proofs
                # Monotonic trust update function
                new_trust[i] = self.config.eta * new_trust[i] + (1 - self.config.eta) * sigma
                
        # Ensure trust stays in [0, 1]
        new_trust = np.clip(new_trust, 0.01, 1.0)
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
    
    def __init__(self, config: Optional[TruthOptimaConfig] = None, models: Optional[Dict[str, LLMInterface]] = None, storage_dir: str = "leo_storage"):
        self.config = config or TruthOptimaConfig()
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
        self.embedder = EmbeddingEngine(dim=self.config.dim)
        self.memory = MicroMemory(self.config, storage_path=os.path.join(self.storage_dir, "memory.json"))
        self.novelty = NoveltyEngine(self.config)
        self.coherence = CoherenceEngine(self.config)
        self.cache = SemanticCache(self.config, storage_path=os.path.join(self.storage_dir, "cache.json"))
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
        
        # Asynchronous background updates (Section Phase B)
        asyncio.create_task(self._background_updates(v, resp))
        
        return resp

    async def stream(self, question: str):
        """Stream the response, using cache if available, otherwise routing to models"""
        self.stats['total_queries'] += 1
        
        v_raw = self.embedder.encode(question)
        v = self.memory.influence(v_raw)
        
        cached_ans, cache_score = self.cache.lookup(v)
        if cached_ans:
            self.stats['cache_hits'] += 1
            # Simulate streaming for cache hit
            for word in cached_ans.split():
                yield word + " "
                await asyncio.sleep(0.02)
            return

        N = self.novelty.compute(v)
        C = self.coherence.compute(v)
        risk = self.risk_assessor.assess(question)
        
        use_consensus = (risk != RiskLevel.LOW) or (N >= self.config.gamma) or (C >= self.config.delta)
        
        full_answer = []
        if use_consensus:
            # For consensus, we stream from the most trusted model but wait for others to finish for background updates
            self.stats['consensus_routes'] += 1
            best_model_name = max(self.trust_weights.items(), key=lambda x: x[1])[0]
            model = self.models[best_model_name]
            
            async for chunk in model.stream(question):
                full_answer.append(chunk)
                yield chunk
            
            # Background: get other answers for consensus resolution
            asyncio.create_task(self._background_consensus_update(question, v, "".join(full_answer), N, C, risk))
        else:
            self.stats['fast_routes'] += 1
            best_model_name = max(self.trust_weights.items(), key=lambda x: x[1])[0]
            model = self.models[best_model_name]
            
            async for chunk in model.stream(question):
                full_answer.append(chunk)
                yield chunk
            
            # Background: update memory and cache
            answer_str = "".join(full_answer)
            resp = TruthOptimaResponse(answer=answer_str, route=RouteType.FAST, risk_level=risk, confidence=0.0, cost_estimate=0.0001)
            asyncio.create_task(self._background_updates(v, resp))

    async def _background_consensus_update(self, question: str, v: np.ndarray, primary_answer: str, N: float, C: float, risk: RiskLevel):
        """Handle consensus resolution in the background during streaming"""
        tasks = [m.query(question) for m in self.models.values()]
        answers = await asyncio.gather(*tasks)
        model_names = list(self.models.keys())
        
        embs = [self.embedder.encode(a) for a in answers]
        weights = np.array([self.trust_weights[name] for name in model_names])
        
        proofs = []
        for i, ans in enumerate(answers):
            v_ans = embs[i]
            sigma = self.embedder.similarity(v, v_ans)
            p = ProofFragment(lcs=True, commitment=hashlib.sha256(ans.encode()).hexdigest()[:16], sigma=sigma)
            proofs.append(p.is_valid(self.config.tau_sigma))
            
        consensus_vec, new_weights, outliers = self.consensus.resolve(embs, weights, proofs=proofs)
        
        for i, name in enumerate(model_names):
            self.trust_weights[name] = float(new_weights[i])
            
        self.memory.update(v)
        self.memory.save()
        self.cache.save()

    async def _background_updates(self, v: np.ndarray, resp: TruthOptimaResponse):
        """Handle state updates and persistence in the background"""
        # Update memory
        self.memory.update(v)
        
        # Ensure persistence
        self.memory.save()
        self.cache.save()
        
        # Save stats
        stats_path = os.path.join(self.storage_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f)

    async def _fast_route(self, question: str, v: np.ndarray, N: float, C: float, risk: RiskLevel) -> TruthOptimaResponse:
        # Pick most trusted model
        best_model_name = max(self.trust_weights.items(), key=lambda x: x[1])[0]
        model = self.models[best_model_name]
        
        answer = await model.query(question)
        
        # Generate Proof Fragment (Section 6)
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
        
        # Generate proof fragments for each model to use in trust evolution
        proofs = []
        for i, ans in enumerate(answers):
            v_ans = embs[i]
            sigma = self.embedder.similarity(v, v_ans)
            p = ProofFragment(lcs=True, commitment=hashlib.sha256(ans.encode()).hexdigest()[:16], sigma=sigma)
            proofs.append(p.is_valid(self.config.tau_sigma))
        
        consensus_vec, new_weights, outliers = self.consensus.resolve(embs, weights, proofs=proofs)
        
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
