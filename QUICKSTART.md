# LEO Optima v2.0 Quick Start Guide

Get up and running with LEO Optima in 5 minutes!

---

## 🚀 Installation (2 minutes)

### Option 1: Docker Compose (Recommended for Production)

This is the easiest way to get LEO Optima running with all its features, including Redis for caching and SQLite for persistent storage.

**Prerequisites:** Ensure you have Docker and Docker Compose installed on your system.

```bash
# 1. Clone the repository
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima

# 2. Configure environment variables
# Copy the example environment file
cp .env.example .env

# IMPORTANT: Edit the .env file to add your API keys.
# Set OPENAI_API_KEY to your OpenAI API key.
# Set LEO_API_KEY to a strong, secret key for securing your LEO Optima proxy.
# Example .env content:
# OPENAI_API_KEY=sk-your-openai-key-here
# LEO_API_KEY=your_secret_leo_api_key

# 3. Build and run the services
docker compose up --build -d
```

LEO Optima will be accessible at `http://localhost:8000`. Redis will be running internally and used automatically.

### Option 2: Manual Installation (for Development/Testing)

```bash
# 1. Clone the repository
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set environment variables
# IMPORTANT: Replace with your actual keys
export OPENAI_API_KEY="sk-your-openai-key-here"
export LEO_API_KEY="your_secret_leo_api_key"

# 4. Start the server
python proxy_server.py
```

**Server should start on `http://localhost:8000`**

---

## 🔌 Integration (1 minute)

LEO Optima acts as a drop-in replacement for your OpenAI API calls. Just change the `base_url` and add the `X-API-Key` header.

### Python + OpenAI Client

```python
from openai import OpenAI
import os

# Point to LEO Optima instead of OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), # Your OpenAI key
    base_url="http://localhost:8000/v1",
    default_headers={
        "X-API-Key": os.getenv("LEO_API_KEY") # Your LEO Optima API Key
    }
)

# Use normally - LEO handles optimization automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
# Access LEO Optima's metrics (if available in your client version)
# print(response.optimization_metrics)
```

### Node.js + OpenAI SDK

```javascript
const OpenAI = require("openai");

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://localhost:8000/v1",
  defaultHeaders: {
    "X-API-Key": process.env.LEO_API_KEY // Your LEO Optima API Key
  }
});

const response = await client.chat.completions.create({
  model: "gpt-4",
  messages: [{ role: "user", content: "Hello!" }]
});

console.log(response.choices[0].message.content);
```

### cURL (with API Key)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d 
  '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is AI?"}]
  }'
```

---

## 📊 Monitor Savings (1 minute)

### Check Analytics (with API Key)

```bash
curl -H "X-API-Key: your_secret_leo_api_key" http://localhost:8000/v1/analytics | jq 
'.'
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

### Enable/Disable Optimizations (with API Key)

```bash
# Disable query decomposition
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d '{"strategy": "query_decomposition", "enabled": false}'

# Re-enable it
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d '{"strategy": "query_decomposition", "enabled": true}'
```

### Check Status (with API Key)

```bash
curl -H "X-API-Key: your_secret_leo_api_key" http://localhost:8000/v1/optimization/status | jq 
'.config'
```

---

## 🎯 Real-World Examples

### Example 1: Simple Query (Cache Hit)

```bash
# First query
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_leo_api_key" \
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
  -H "X-API-Key: your_secret_leo_api_key" \
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
  -H "X-API-Key: your_secret_leo_api_key" \
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
  -H "X-API-Key: your_secret_leo_api_key" \
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

### Server Won't Start (Docker Compose)
1.  **Check Docker Status**: Ensure Docker Desktop (or daemon) is running.
2.  **Verify `.env`**: Make sure `OPENAI_API_KEY` and `LEO_API_KEY` are correctly set in your `.env` file.
3.  **Inspect Logs**: Run `docker compose logs leo-optima` to see detailed error messages from the application container.
4.  **Rebuild**: Try `docker compose down --volumes && docker compose up --build -d` to ensure a clean build.

### Server Won't Start (Manual)

```bash
# Check Python version
python --version  # Should be 3.11+

# Check dependencies
pip install -r requirements.txt

# Verify environment variables are set
echo $OPENAI_API_KEY
echo $LEO_API_KEY

# Try verbose mode
python proxy_server.py --debug
```

### Low Cache Hit Rate

```bash
# Check cache stats
curl -H "X-API-Key: your_secret_leo_api_key" http://localhost:8000/v1/optimization/cache/stats

# Provide feedback to improve threshold
curl -X POST http://localhost:8000/v1/optimization/cache/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d '{
    "cached_answer": "...",
    "is_correct": true
  }'
```

### High API Costs Still

```bash
# Check which optimizations are active
curl -H "X-API-Key: your_secret_leo_api_key" http://localhost:8000/v1/optimization/status

# Enable all optimizations
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d '{"strategy": "adaptive_cache", "enabled": true}'
```

---

## 📚 Next Steps

1.  **Read Full Documentation**: [README.md](README.md)
2.  **API Reference**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
3.  **Technical Details**: [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)

---

## 💡 Pro Tips

### Tip 1: Monitor in Real-time

```bash
# Watch analytics every 5 seconds
watch -n 5 'curl -s -H "X-API-Key: your_secret_leo_api_key" http://localhost:8000/v1/analytics | jq ".optimization_metrics"'
```

### Tip 2: Test with Different Queries

```bash
# Simple query (should cache hit quickly)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'

# Complex query (should decompose)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Compare A and B, explain C, then summarize"}]}'
```

### Tip 3: Use Feedback Loop

```bash
# After each response, provide feedback
curl -X POST http://localhost:8000/v1/optimization/cache/feedback \
  -H "X-API-Key: your_secret_leo_api_key" \
  -d '{"cached_answer":"...","is_correct":true}'

# This improves cache threshold over time
```

---

## 🎯 Success Metrics

Track these to measure success:

```bash
# Weekly check
curl -H "X-API-Key: your_secret_leo_api_key" http://localhost:8000/v1/analytics | jq '{
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
