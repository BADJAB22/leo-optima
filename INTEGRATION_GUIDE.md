# LEO Optima: Integration & Technical Guide

LEO Optima is a high-performance routing and optimization engine that sits between your application and various LLM providers. It uses semantic analysis, novelty detection, and Byzantine consensus to ensure cost-efficiency and response reliability.

---

## Key Features (Phase A, B & C Implemented)

### 1. Lightweight Semantic Foundation
We have implemented a high-performance **Johnson-Lindenstrauss Projection** based embedding engine. This allows for:
- **Zero-Dependency Semantics**: Stable vector representations without needing heavy models like BERT or Sentence-Transformers.
- **Fast Similarity**: Pure `numpy` implementation for sub-millisecond cache lookups and novelty detection.

### 2. Verifiable Optimization Proofs (Section 6)
Each response now includes a `ProofFragment` object:
- **LCS (Local Constraint Satisfaction)**: Ensures the optimization was performed within cost/resource constraints.
- **Commitment Hash**: A cryptographic hash `H(cost_opt)` that commits the model to its optimized state.
- **Sigma (Semantic Consistency)**: Measures `sim(answer, query)` to ensure the response hasn't drifted semantically.

### 3. ADMM-based Coherence Engine (Section 4)
The Coherence Engine now uses a refined **Alternating Direction Method of Multipliers (ADMM)** to calculate the consensus stability of a query. 
- High coherence ($C < \delta$) indicates a stable, well-understood query.
- Low coherence ($C \ge \delta$) triggers the **CONSENSUS** route for higher reliability.

### 4. Cache Poisoning Mitigation
The `SemanticCache` now includes **Trust Gating**. Cached answers are weighted by the trust score of the model that generated them. If a model's trust falls below a threshold, its cached entries are effectively ignored, preventing adversarial "poisoning" of the semantic cache.

---

## API Integration

### Routing Response Object
When calling `TruthOptima.ask()`, you receive a `TruthOptimaResponse` containing:
| Field | Type | Description |
| :--- | :--- | :--- |
| `answer` | `str` | The generated response. |
| `route` | `RouteType` | `CACHE`, `FAST`, or `CONSENSUS`. |
| `proof` | `ProofFragment` | The verifiable fragment for audit logs. |
| `confidence`| `float` | The system's internal confidence score. |

### Example Usage
```python
from Truth_Optima import TruthOptima

# Initialize the engine
system = TruthOptima()

# Process a query
response = await system.ask("Is aspirin safe during pregnancy?")

# Verify the proof
if response.proof.is_valid(tau_sigma=0.87):
    print(f"Verified Answer: {response.answer}")
```

---

## Technical Specifications (Current)
- **Embedding Dim**: 384
- **Memory Max**: 32 vectors
- **Cache Max**: 2000 entries
- **Consensus**: Byzantine-robust weighted mean with outlier detection.
- **Cache Poisoning Mitigation**: Implemented via trust gating in `SemanticCache`.
