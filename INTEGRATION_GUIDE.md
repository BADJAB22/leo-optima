# 🚀 LEO Optima: One-Click Integration Guide

LEO Optima works as a smart middleware between your application and any LLM provider. By changing just one line of code, you can start saving up to 80% on API costs.

## 1. Start the LEO Proxy
Run the following command to start the optimization engine on your server:
```bash
python3 proxy_server.py
```
*The proxy runs on `http://localhost:8000` by default.*

## 2. Update Your Code
Simply change the `base_url` in your LLM client to point to the LEO Proxy.

### Python (OpenAI Library)
```python
from openai import OpenAI

# OLD CODE
# client = OpenAI(api_key="your_key")

# NEW OPTIMIZED CODE
client = OpenAI(
    api_key="your_key", 
    base_url="http://localhost:8000/v1" # Just add this line!
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Your query here"}]
)
```

### Node.js (OpenAI SDK)
```javascript
const { OpenAI } = require('openai');

const openai = new OpenAI({
  apiKey: 'your_key',
  baseURL: 'http://localhost:8000/v1' // Just add this line!
});
```

## 3. How it Works
1. **Semantic Check:** LEO analyzes the query's meaning.
2. **Instant Hit:** If a similar query was asked before, LEO returns the answer in **<10ms** for **$0**.
3. **Smart Pass:** If it's a new query, LEO forwards it to the real provider, then caches the result for future use.

## 4. Key Benefits
- **Universal:** Works with OpenAI, Anthropic, Groq, or Local LLMs.
- **Persistent:** The cache is saved to `leo_cache.json` and survives restarts.
- **Privacy First:** All optimization happens on your infrastructure.
