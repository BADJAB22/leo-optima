# LEO Optima v2.0 Quick Start Guide

Get up and running with LEO Optima in 5 minutes!

---

## 🚀 Installation (2 minutes)

### Option 1: Direct Installation

```bash
# Clone repository
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima

# Install dependencies
pip install -r requirements.txt

# Start server
python proxy_server.py
```

### Option 2: Docker (Recommended)

```bash
# Build image
docker build -t leo-optima:v2 .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  leo-optima:v2
```

**Server should start on `http://localhost:8000`**

---

## 🔌 Integration (1 minute)

### Python + OpenAI Client

```python
from openai import OpenAI

# Change just one line!
client = OpenAI(
    api_key="your-openai-key",
    base_url="http://localhost:8000/v1"  # ← Point to LEO
)

# Use normally
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Node.js + OpenAI SDK

```javascript
const OpenAI = require('openai');

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: 'http://localhost:8000/v1'  // ← Point to LEO
});

const response = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is AI?"}]
  }'
```

---

## 📊 Monitor Savings (1 minute)

### Check Analytics

```bash
curl http://localhost:8000/v1/analytics | jq '.'
```

**Look for:**
- `cache_hit_rate`: % of requests served from cache
- `cost_saved`: Total $ saved
- `tokens_saved`: Total tokens reduced
- `average_confidence`: Quality of responses

### Example Output

```json
{
  "optimization_metrics": {
    "total_requests": 1250,
    "cache_hits": 450,
    "cache_hit_rate": 0.36,
    "cost_saved": 45.50,
    "tokens_saved": 12500,
    "average_confidence": 0.87
  }
}
```

---

## ⚙️ Configuration (1 minute)

### Enable/Disable Optimizations

```bash
# Disable query decomposition
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -d '{"strategy": "query_decomposition", "enabled": false}'

# Re-enable it
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -d '{"strategy": "query_decomposition", "enabled": true}'
```

### Check Status

```bash
curl http://localhost:8000/v1/optimization/status | jq '.config'
```

---

## 🎯 Real-World Examples

### Example 1: Simple Query (Cache Hit)

```bash
# First query
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }' | jq '.optimization_metrics'

# Output:
# {
#   "cache_hit": false,
#   "tokens_saved": 12,
#   "processing_time_ms": 1245.5
# }

# Second query (identical)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }' | jq '.optimization_metrics'

# Output:
# {
#   "cache_hit": true,
#   "tokens_saved": 0,
#   "processing_time_ms": 2.5  ← 500x faster!
# }
```

### Example 2: Complex Query (Decomposition)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{
      "role": "user",
      "content": "Compare Python and JavaScript, then explain their use cases"
    }]
  }' | jq '.optimization_metrics'

# Output:
# {
#   "decomposed": true,  ← Query was broken into parts
#   "tokens_saved": 45,
#   "processing_time_ms": 800
# }
```

### Example 3: Confidence Scoring

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Explain quantum computing"}]
  }' | jq '.optimization_metrics'

# Output:
# {
#   "confidence_score": 0.87,  ← High confidence
#   "tokens_saved": 23,
#   "processing_time_ms": 1100
# }
```

---

## 📈 Expected Results

### After 1 Week

- Cache size: 50-100 entries
- Cache hit rate: 10-15%
- Cost savings: 5-10%

### After 1 Month

- Cache size: 200-500 entries
- Cache hit rate: 25-35%
- Cost savings: 40-50%

### After 3 Months

- Cache size: 500-1000 entries
- Cache hit rate: 35-45%
- Cost savings: 60-75%

---

## 🔧 Troubleshooting

### Server Won't Start

```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip install -r requirements.txt

# Try verbose mode
python proxy_server.py --debug
```

### Low Cache Hit Rate

```bash
# Check cache stats
curl http://localhost:8000/v1/optimization/cache/stats

# Provide feedback to improve threshold
curl -X POST http://localhost:8000/v1/optimization/cache/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "cached_answer": "...",
    "is_correct": true
  }'
```

### High API Costs Still

```bash
# Check which optimizations are active
curl http://localhost:8000/v1/optimization/status

# Enable all optimizations
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -d '{"strategy": "adaptive_cache", "enabled": true}'
```

---

## 📚 Next Steps

1. **Read Full Documentation**: [README.md](README.md)
2. **API Reference**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
3. **Technical Details**: [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)
4. **Single Model Guide**: [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)

---

## 💡 Pro Tips

### Tip 1: Monitor in Real-time

```bash
# Watch analytics every 5 seconds
watch -n 5 'curl -s http://localhost:8000/v1/analytics | jq ".optimization_metrics"'
```

### Tip 2: Test with Different Queries

```bash
# Simple query (should cache hit quickly)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'

# Complex query (should decompose)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Compare A and B, explain C, then summarize"}]}'
```

### Tip 3: Use Feedback Loop

```bash
# After each response, provide feedback
curl -X POST http://localhost:8000/v1/optimization/cache/feedback \
  -d '{"cached_answer":"...","is_correct":true}'

# This improves cache threshold over time
```

---

## 🎯 Success Metrics

Track these to measure success:

```bash
# Weekly check
curl http://localhost:8000/v1/analytics | jq '{
  cache_hit_rate: .optimization_metrics.cache_hit_rate,
  cost_saved: .optimization_metrics.cost_saved,
  tokens_saved: .optimization_metrics.tokens_saved,
  avg_confidence: .optimization_metrics.average_confidence
}'
```

**Target values after 1 month:**
- Cache hit rate: > 25%
- Cost saved: > $50
- Tokens saved: > 10,000
- Avg confidence: > 0.85

---

## 🚀 You're Ready!

LEO Optima is now running and optimizing your LLM costs. Start seeing savings immediately!

**Questions?** Check the [full documentation](README.md) or open an issue.

**Happy optimizing!** 🎉
