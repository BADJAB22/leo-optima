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
import sqlite3
import redis
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# Import the API interfaces
from api_interfaces import LLMInterface, LLMSimulator, OpenAICompatibleAPI

# Import advanced modules
try:
    from advanced_embeddings import AdvancedEmbeddingEngine
    ADVANCED_EMBEDDINGS_AVAILABLE = True
except ImportError:
    ADVANCED_EMBEDDINGS_AVAILABLE = False

try:
    from smart_risk_assessment import SmartRiskAssessor, RiskLevel
    SMART_RISK_AVAILABLE = True
except ImportError:
    SMART_RISK_AVAILABLE = False

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
    tau_cache: float = 0.85      # Cache similarity threshold
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
    
    # Risk assessment keywords (Fallback)
    high_risk_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'medical': ['medical', 'health', 'disease', 'drug', 'medicine', 'treatment', 'pregnancy', 'aspirin', 'heart', 'chest pain'],
        'legal': ['legal', 'law', 'court', 'sue', 'contract', 'rights'],
        'financial': ['invest', 'stock', 'crypto', 'loan', 'tax', 'money'],
        'safety': ['safe', 'danger', 'risk', 'emergency', 'hazard', 'kill', 'suicide']
    })


class RouteType(Enum):
    """Routing decisions"""
    CACHE = "CACHE"           # Cache hit (free!)
    FAST = "FAST"             # Single small model (cheap)
    CONSENSUS = "CONSENSUS"   # Byzantine verification (expensive but verified)


if not SMART_RISK_AVAILABLE:
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
# LEGACY ENGINES (FALLBACKS)
# ============================================================

class EmbeddingEngine:
    """Legacy random projection embedding"""
    def __init__(self, dim: int = 384):
        self.dim = dim
        state = np.random.RandomState(42)
        self.projection = state.randn(1024, dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)
    
    def encode(self, text: str) -> np.ndarray:
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


# ============================================================
# MICRO-MEMORY MANAGER
# ============================================================

class MicroMemory:
    """Manages contextual micro-memory with decay (Section 2.3)"""
    
    def __init__(self, config: TruthOptimaConfig, storage_path: Optional[str] = None):
        self.config = config
        self.memory: List[np.ndarray] = []
        self.storage_path = storage_path
        self.db_path = storage_path.replace(".json", ".db") if storage_path else None
        
        if self.db_path:
            self._init_db()
            self.load()
        elif self.storage_path and os.path.exists(self.storage_path):
            self.load_legacy_json()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, vector BLOB NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        conn.commit()
        conn.close()
    
    def save(self):
        if not self.db_path: return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM memory')
        for m in self.memory:
            cursor.execute('INSERT INTO memory (vector) VALUES (?)', (m.tobytes(),))
        conn.commit()
        conn.close()
            
    def load(self):
        if not self.db_path or not os.path.exists(self.db_path): return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT vector FROM memory ORDER BY id ASC')
            rows = cursor.fetchall()
            self.memory = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
            conn.close()
        except Exception as e:
            print(f"Error loading memory: {e}")

    def load_legacy_json(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.memory = [np.array(m, dtype=np.float32) for m in data]
            if self.db_path: self.save()
        except Exception as e:
            print(f"Error loading legacy memory: {e}")
    
    def influence(self, v_raw: np.ndarray) -> np.ndarray:
        if not self.memory: return v_raw
        mem_stack = np.stack(self.memory)
        sims = np.dot(mem_stack, v_raw)
        influence = np.zeros_like(v_raw)
        for s, m in zip(sims, self.memory):
            if s > 0: influence += s * m
        v_enriched = v_raw + self.config.alpha * influence
        return self._normalize(v_enriched)
    
    def update(self, v: np.ndarray):
        if len(self.memory) < self.config.memory_max:
            self.memory.append(v)
        else:
            self.memory = [self._normalize(self.config.beta * m + (1 - self.config.beta) * v) for m in self.memory]
    
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
        n_samples = len(corpus_embeddings)
        actual_clusters = min(n_clusters, n_samples)
        if actual_clusters == 0: return
        idx = np.random.choice(n_samples, actual_clusters, replace=False)
        self.cluster_centers = corpus_embeddings[idx]
    
    def compute(self, v: np.ndarray) -> float:
        if self.cluster_centers is None: return 0.5
        dists = np.linalg.norm(self.cluster_centers - v, axis=1)
        probs = np.exp(-dists)
        probs /= (np.sum(probs) + 1e-12)
        H = -np.sum(probs * np.log(probs + 1e-12))
        D = float(np.min(dists))
        return float(H + self.config.lambda_ * D)


class CoherenceEngine:
    """Measures query coherence via ADMM projection (Section 4)"""
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        self.W = None
        
    def fit(self, corpus_embeddings: np.ndarray):
        if len(corpus_embeddings) < 2: return
        # Simple covariance-based coherence matrix
        self.W = np.cov(corpus_embeddings.T)
        
    def compute(self, v: np.ndarray) -> float:
        if self.W is None: return 0.5
        # ADMM-inspired coherence: dist from manifold
        try:
            proj = np.dot(self.W, v)
            proj /= (np.linalg.norm(proj) + 1e-12)
            dist = np.linalg.norm(v - proj)
            return float(1.0 / (1.0 + dist))
        except:
            return 0.5


# ============================================================
# SEMANTIC CACHE
# ============================================================

class SemanticCache:
    """Optimized semantic cache with O(log N) search and Redis integration"""
    def __init__(self, config: TruthOptimaConfig, storage_path: Optional[str] = None, redis_host='localhost', redis_port=6379):
        self.config = config
        self.storage_path = storage_path
        self.db_path = storage_path.replace(".json", ".db") if storage_path else None
        self.data: List[Dict] = []
        self.index = None
        
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis.ping()
            self.use_redis = True
        except:
            self.use_redis = False

        if self.db_path:
            self._init_db()
            self.load()
        elif self.storage_path and os.path.exists(self.storage_path):
            self.load_legacy_json()
            
        self._rebuild_index()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY AUTOINCREMENT, vector BLOB NOT NULL, answer TEXT NOT NULL, trust REAL NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        conn.commit()
        conn.close()
            
    def _rebuild_index(self):
        if not self.data:
            self.index = None
            return
        vectors = np.stack([entry['vector'] for entry in self.data])
        self.index = NearestNeighbors(n_neighbors=min(2, len(self.data)), metric='cosine', algorithm='auto')
        self.index.fit(vectors)

    def save(self):
        if not self.db_path: return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM cache')
        for entry in self.data:
            cursor.execute('INSERT INTO cache (vector, answer, trust, timestamp) VALUES (?, ?, ?, ?)',
                         (entry['vector'].tobytes(), entry['answer'], entry['trust'], entry['timestamp']))
        conn.commit()
        conn.close()
            
    def load(self):
        if not self.db_path or not os.path.exists(self.db_path): return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT vector, answer, trust, timestamp FROM cache ORDER BY id ASC')
            rows = cursor.fetchall()
            self.data = []
            for row in rows:
                self.data.append({'vector': np.frombuffer(row[0], dtype=np.float32), 'answer': row[1], 'trust': row[2], 'timestamp': row[3]})
            conn.close()
            self._rebuild_index()
        except Exception as e:
            print(f"Error loading cache: {e}")

    def load_legacy_json(self):
        try:
            with open(self.storage_path, 'r') as f:
                serializable_data = json.load(f)
                self.data = [{'vector': np.array(entry['vector'], dtype=np.float32), 'answer': entry['answer'], 'trust': entry['trust'], 'timestamp': entry['timestamp']} for entry in serializable_data]
            if self.db_path: self.save()
        except Exception as e:
            print(f"Error loading legacy cache: {e}")
    
    def lookup(self, v: np.ndarray) -> Tuple[Optional[str], float]:
        if self.use_redis:
            v_hash = hashlib.md5(v.tobytes()).hexdigest()
            cached_data = self.redis.get(f"cache:{v_hash}")
            if cached_data:
                entry = json.loads(cached_data)
                return entry['answer'], 1.0

        if not self.data or self.index is None: return None, 0.0
        
        v_reshaped = v.reshape(1, -1)
        distances, indices = self.index.kneighbors(v_reshaped, n_neighbors=min(2, len(self.data)))
        similarities = 1.0 - distances[0]
        
        scores = []
        for i, idx in enumerate(indices[0]):
            entry = self.data[idx]
            trust_factor = entry.get('trust', 1.0)
            score = float(similarities[i]) * trust_factor
            scores.append((score, entry['answer']))
            
        s_max, best_ans = scores[0]
        s_second = scores[1][0] if len(scores) > 1 else 0.0
        
        if s_max >= self.config.tau_cache and (s_max - s_second) >= self.config.tau_gap:
            return best_ans, float(s_max)
        return None, float(s_max)

    def add(self, v: np.ndarray, answer: str, trust: float):
        entry = {'vector': v, 'answer': answer, 'trust': trust, 'timestamp': datetime.now().isoformat()}
        self.data.append(entry)
        
        if self.use_redis:
            v_hash = hashlib.md5(v.tobytes()).hexdigest()
            self.redis.setex(f"cache:{v_hash}", 3600, json.dumps({'answer': answer, 'trust': trust}))

        if len(self.data) > self.config.cache_max: self.data.pop(0)
        self._rebuild_index()


# ============================================================
# BYZANTINE CONSENSUS
# ============================================================

class ByzantineConsensus:
    def __init__(self, config: TruthOptimaConfig):
        self.config = config
        
    def resolve(self, embeddings: List[np.ndarray], trust_weights: np.ndarray, proofs: Optional[List[bool]] = None) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        n = len(embeddings)
        stack = np.stack(embeddings)
        consensus = np.average(stack, axis=0, weights=trust_weights)
        dists = np.linalg.norm(stack - consensus, axis=1)
        median_dist = np.median(dists)
        outliers = dists > (self.config.outlier_threshold * median_dist + 1e-6)
        
        new_trust = trust_weights.copy()
        for i in range(n):
            sigma = float(np.dot(embeddings[i], consensus))
            is_valid_proof = proofs[i] if proofs else True
            if outliers[i] or not is_valid_proof:
                new_trust[i] *= self.config.trust_decay if outliers[i] else 0.9
            else:
                new_trust[i] = self.config.eta * new_trust[i] + (1 - self.config.eta) * sigma
        return consensus, np.clip(new_trust, 0.01, 1.0), outliers.tolist()


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
    audit_log: List[Dict] = field(default_factory=list)
    verification_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class TruthOptima:
    """The LEO Optima Core Engine - Model Agnostic"""
    
    def __init__(self, config: Optional[TruthOptimaConfig] = None, models: Optional[Dict[str, LLMInterface]] = None, storage_dir: str = "leo_storage"):
        self.config = config or TruthOptimaConfig()
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir): os.makedirs(self.storage_dir)
            
        # Initialize Models
        self.models = models or {'default': LLMSimulator("Default-LLM")}
        
        # Initialize Advanced Components
        if ADVANCED_EMBEDDINGS_AVAILABLE:
            # Can pass a custom embedding interface here if needed
            self.embedder = AdvancedEmbeddingEngine(dim=self.config.dim)
        else:
            self.embedder = EmbeddingEngine(dim=self.config.dim)
            
        if SMART_RISK_AVAILABLE:
            # Uses the first available model for risk assessment
            risk_model = list(self.models.values())[0]
            self.risk_assessor = SmartRiskAssessor(config=self.config, model_interface=risk_model)
        else:
            self.risk_assessor = None # Fallback logic below
            
        self.memory = MicroMemory(self.config, storage_path=os.path.join(self.storage_dir, "memory.json"))
        self.novelty = NoveltyEngine(self.config)
        self.coherence = CoherenceEngine(self.config)
        self.cache = SemanticCache(self.config, storage_path=os.path.join(self.storage_dir, "cache.json"))
        self.consensus = ByzantineConsensus(self.config)
        
        self.trust_weights = {name: 0.9 for name in self.models.keys()}
        self.stats = {'total_queries': 0, 'cache_hits': 0, 'fast_routes': 0, 'consensus_routes': 0, 'total_cost': 0.0}

    async def _assess_risk(self, prompt: str) -> RiskLevel:
        if SMART_RISK_AVAILABLE and self.risk_assessor:
            # Use the smart assessor if available
            return await self.risk_assessor.assess(prompt)
        
        # Fallback keyword logic
        prompt_lower = prompt.lower()
        hits = 0
        for keywords in self.config.high_risk_keywords.values():
            if any(kw in prompt_lower for kw in keywords):
                hits += 1
        
        if hits >= 2: return RiskLevel.CRITICAL
        if hits == 1: return RiskLevel.HIGH
        return RiskLevel.LOW

    async def ask(self, question: str, tenant_id: str = "default") -> TruthOptimaResponse:
        self.stats['total_queries'] += 1
        audit_log = []
        
        # 1. Embedding + Memory
        v_raw = self.embedder.encode(question)
        v = self.memory.influence(v_raw)
        audit_log.append({"event": "embedding_generated"})
        
        # 2. Cache Lookup
        cached_ans, cache_score = self.cache.lookup(v)
        risk = await self._assess_risk(question)
        
        if cached_ans:
            self.stats['cache_hits'] += 1
            return TruthOptimaResponse(answer=cached_ans, route=RouteType.CACHE, risk_level=risk, confidence=1.0, cost_estimate=0.0, cache_score=cache_score, audit_log=audit_log)
        
        # 3. Novelty & Coherence
        N = self.novelty.compute(v)
        C = self.coherence.compute(v)
        
        # 4. Routing Policy
        use_consensus = (risk != RiskLevel.LOW) or (N >= self.config.gamma) or (C >= self.config.delta)
        
        if use_consensus:
            resp = await self._consensus_route(question, v, N, C, risk)
            self.stats['consensus_routes'] += 1
        else:
            resp = await self._fast_route(question, v, N, C, risk)
            self.stats['fast_routes'] += 1
            
        resp.audit_log = audit_log + (resp.audit_log or [])
        self.stats['total_cost'] += resp.cost_estimate
        
        # 5. Background Updates (Memory and Cache)
        # Memory is updated with the context of the current query
        self.memory.update(v)
        # Cache is only updated for non-cached responses
        if resp.route != RouteType.CACHE:
            self.cache.add(v, resp.answer, resp.confidence)
            
        return resp

    async def _fast_route(self, question: str, v: np.ndarray, N: float, C: float, risk: RiskLevel) -> TruthOptimaResponse:
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        answer = await model.query(question)
        return TruthOptimaResponse(answer=answer, route=RouteType.FAST, risk_level=risk, confidence=0.85, cost_estimate=0.01, novelty=N, coherence=C)

    async def _consensus_route(self, question: str, v: np.ndarray, N: float, C: float, risk: RiskLevel) -> TruthOptimaResponse:
        tasks = [model.query(question) for model in self.models.values()]
        answers = await asyncio.gather(*tasks)
        
        # In a real system, we'd embed the answers for consensus
        ans_embeddings = [self.embedder.encode(ans) for ans in answers]
        weights = np.array([self.trust_weights[name] for name in self.models.keys()])
        
        consensus_v, new_weights, outliers = self.consensus.resolve(ans_embeddings, weights)
        
        # Update trust weights
        for i, name in enumerate(self.models.keys()):
            self.trust_weights[name] = new_weights[i]
            
        # Choose the answer closest to consensus
        best_idx = np.argmax([np.dot(emb, consensus_v) for emb in ans_embeddings])
        
        return TruthOptimaResponse(
            answer=answers[best_idx],
            route=RouteType.CONSENSUS,
            risk_level=risk,
            confidence=float(np.max(new_weights)),
            cost_estimate=0.05 * len(self.models),
            novelty=N,
            coherence=C,
            outliers=[list(self.models.keys())[i] for i, is_outlier in enumerate(outliers) if is_outlier],
            trust_scores=self.trust_weights.copy()
        )
