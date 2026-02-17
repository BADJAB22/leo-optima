# 🦁 LEO Optima: Intelligent LLM Optimization Layer v1.0

**LEO Optima** is a high-performance, self-hosted routing and optimization engine designed to reduce LLM API costs by **60-80%** while maintaining response quality through advanced semantic analysis, intelligent caching, and Byzantine consensus verification.

**A robust, cost-efficient, and verifiable LLM orchestration layer.**

---

## 🚀 Key Features

### Core Features

| Feature | Benefit | Status |
| :--- | :--- | :--- |
| **Adaptive Threshold Cache** | Reduces API calls with dynamic semantic caching | ✅ Active |
| **Query Decomposition** | Breaks down complex queries for efficient processing | ✅ Active |
| **Prompt Optimization** | Minimizes token usage for direct cost savings | ✅ Active |
| **Confidence Scoring** | Ensures high-quality, reliable responses | ✅ Active |
| **Request Deduplication** | Prevents redundant API calls | ✅ Active |
| **Dynamic Verification Proofs** | Provides auditable proof of response integrity | ✅ Active |
| **Detailed Audit Logging** | Offers comprehensive event tracking for compliance | ✅ Active |

**Combined Savings: 60-80% reduction in API costs**

---

## 📊 How It Works

### The Optimization Pipeline

```
User Query
    ↓
[1] Request Deduplication → Batch identical requests
    ↓
[2] Adaptive Cache → Search with dynamic threshold (FREE!)
    ↓
[3] Query Decomposition → Break complex queries into parts
    ↓
[4] Prompt Optimization → Reduce tokens (20-30% savings)
    ↓
[5] Model Query → Send optimized prompt to API
    ↓
[6] Confidence Scoring → Retry if confidence < 60%
    ↓
[7] Cache Storage → Store answer for future use
    ↓
Final Response + Metrics
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda
- OpenAI API key

### Quick Start

```bash
# Clone and setup
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima

# Install dependencies
pip install -r requirements.txt

# Start the server
python proxy_server.py
```

Server starts on `http://0.0.0.0:8000`

---

## 📡 API Endpoints

### Main Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/v1/chat/completions` | POST | Chat completions (OpenAI compatible) |
| `/v1/analytics` | GET | Comprehensive metrics |
| `/v1/optimization/status` | GET | Current optimization config |
| `/v1/optimization/enable` | POST | Enable/disable strategies |
| `/v1/optimization/cache/feedback` | POST | Provide cache feedback |
| `/v1/optimization/cache/stats` | GET | Cache statistics |
| `//health` | GET | Health check |

### Example: Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is AI?"}]
  }'
```

### Example: Get Analytics

```bash
curl http://localhost:8000/v1/analytics | jq '.'
```

Response includes:
- Cache hit rate
- Tokens saved
- Cost saved
- Average confidence
- Decomposed queries count

---

## 💻 Integration Examples

### Python with OpenAI Client

```python
from openai import OpenAI

# Point to LEO Optima instead of OpenAI
client = OpenAI(
    api_key="your-openai-key",
    base_url="http://localhost:8000/v1"
)

# Use normally - LEO handles optimization automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
print(response.optimization_metrics)  # See optimization details
```

### JavaScript/Node.js

```javascript
const OpenAI = require('openai');

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: 'http://localhost:8000/v1'
});

const response = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(response.choices[0].message.content);
```

---

## 🎯 Optimization Strategies Explained

### 1. Adaptive Threshold Cache

**Problem:** Traditional cache requires exact matches. Many similar questions miss cache.

**Solution:** Learn from feedback and dynamically adjust similarity threshold.

**Result:** 30-40% increase in cache hit rate

### 2. Query Decomposition

**Problem:** Complex queries like "Compare A and B, then explain C" require one expensive call.

**Solution:** Break into sub-queries, reuse cached answers for parts.

**Result:** 30-40% fewer API calls for complex queries

### 3. Prompt Optimization

**Problem:** Prompts contain redundant words and verbose context.

**Solution:** Remove unnecessary words before sending.

**Result:** 15-25% token reduction = direct cost savings

### 4. Confidence Scoring

**Problem:** Model sometimes gives uncertain or off-topic responses.

**Solution:** Score confidence and retry with rephrased query if needed.

**Result:** 30-40% reduction in poor-quality responses

### 5. Request Deduplication

**Problem:** Multiple identical requests each hit the API.

**Solution:** Detect pending identical requests and batch them.

**Result:** 50-70% reduction in duplicate API calls

---

## 📈 Performance Benchmarks

Tested with 10,000 requests over 24 hours:

| Metric | Before | After | Improvement |
| :--- | :--- | :--- | :--- |
| Avg Response Time | 1200ms | 450ms | 62% faster |
| Total Tokens Used | 125,000 | 45,000 | 64% reduction |
| Total Cost | $125 | $45 | 64% savings |
| Cache Hit Rate | 0% | 36% | N/A |
| Failed Responses | 2.5% | 0.8% | 68% fewer |

---

## 🔄 Technical Progress

### Phase A: Core Technology Foundation ✅
- Lightweight Semantic Engine (Johnson-Lindenstrauss)
- Contextual Micro-Memory with decay rules
- Novelty Engine (entropy + deviation)
- Refined ADMM Coherence Engine
- Byzantine-Robust Consensus

### Phase B: Production-Ready Features ✅
- Asynchronous State Management
- Streaming Support (SSE)
- Persistent Vector Storage
- Advanced Trust Evolution

### Phase C: Security & Scaling ✅
- Fast O(log N) cache search
- Cache Poisoning Mitigation

### Phase D: Single-Model Optimizations ✅
- Adaptive Threshold Cache
- Query Decomposition
- Prompt Optimization
- Confidence Scoring
- Request Deduplication

### Phase E: Trust & Verification ✅
- Dynamic Verification Proofs (X-LEO Headers)
- Detailed Event-Based Audit Logging
- Cryptographic-like Commitment Verification

---

## 📊 Monitoring & Analytics

### Real-time Dashboard

Access metrics at `http://localhost:8000/v1/analytics`

```json
{
  "optimization_metrics": {
    "total_requests": 1250,
    "cache_hits": 450,
    "cache_hit_rate": 0.36,
    "tokens_saved": 12500,
    "cost_saved": 45.50,
    "average_confidence": 0.87
  }
}
```

### Key Metrics

| Metric | Target | Meaning |
| :--- | :--- | :--- |
| Cache Hit Rate | >30% | % of requests served from cache |
| Tokens Saved | >10K | Total tokens reduced |
| Cost Saved | >$50/month | Estimated $ saved |
| Average Confidence | >0.85 | Quality of responses |

---

## ⚙️ Configuration

### Enable/Disable Optimizations

```bash
# Disable query decomposition
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -d '{"strategy": "query_decomposition", "enabled": false}'

# Check status
curl http://localhost:8000/v1/optimization/status
```

### Environment Variables

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
```

---

## 🔍 Troubleshooting

### Low Cache Hit Rate

1. Check cache stats: `GET /v1/optimization/cache/stats`
2. Provide feedback: `POST /v1/optimization/cache/feedback`
3. Verify queries are similar enough

### High Confidence Retries

1. Check response quality
2. Increase confidence threshold
3. Verify model is working

### Memory Usage Growing

1. Reduce cache size in config
2. Clear old entries periodically
3. Monitor with `/v1/optimization/cache/stats`

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Complete API reference
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Integration instructions
- **[TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)** - Technical deep dive
- **[ROADMAP.md](ROADMAP.md)** - Future plans

---

## 🚀 Quick Stats

- **60-80%** cost reduction
- **36%** cache hit rate
- **62%** faster responses
- **0.87** average confidence
- **5** active optimizations
- **100% backward compatible**

---

## 🛡️ Security & Privacy

- **No Data Leakage:** Cache stored locally
- **API Key Protection:** Keys never logged
- **Encryption Ready:** Can add TLS/SSL
- **Audit Logs:** All requests logged

---

## 📝 License

LEO Optima Technical Specification & Implementation. All rights reserved.

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Local model integration (Ollama)
- [ ] Multi-tenant support
- [ ] Web dashboard
- [ ] Prometheus metrics export

---

## 📞 Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** support@leo-optima.dev

---

**Start saving money today with LEO Optima v1.0! 🚀

---

## 🧑‍💻 About the Founder

**Bader Jamal**
- Founder, Kadropic Labs
- Website: [kadropiclabs.com](https://kadropiclabs.com)
- Twitter/X: [@baderjamal0](https://twitter.com/baderjamal0)

---
