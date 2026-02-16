# 🦁 LEO Optima: The Intelligent LLM Optimization Layer

**LEO Optima** is a high-performance, self-hosted middleware designed to slash LLM API costs by up to 80% while significantly reducing latency. It acts as a "Smart Filter" between your application and any AI provider (OpenAI, Anthropic, Groq, or Local LLMs).

---

## 🎯 The Core Philosophy
In the current AI landscape, companies pay full price for every single token, even when asking the same or similar questions repeatedly. **LEO Optima changes this.** 

We don't provide the models; **you bring your own models.** We provide the **Intelligence Layer** that ensures you never pay for the same answer twice.

---

## ✨ Key Features

### 🧠 Advanced Semantic Caching
Unlike traditional keyword-based caches, LEO uses **Vector Embeddings** to understand the *meaning* of a query. If a user asks "What's the weather in London?" and another asks "Tell me the current climate in London," LEO recognizes the similarity and serves the cached result instantly.

### 🛡️ Privacy-First Architecture (Self-Hosted)
- **Zero-Data Leakage:** LEO runs on **your** infrastructure. Your queries, API keys, and cached data never leave your servers.
- **Local Vector DB:** Uses a high-speed local storage system for embeddings, ensuring total control over your intellectual property.

### ⚡ Ultra-Low Latency
- **Instant Hits:** Cached responses are served in **<10ms**, compared to the 2-10 seconds typical of raw LLM calls.
- **Efficient Routing:** Non-cached queries are forwarded to the provider with minimal overhead.

### 🔌 Universal Compatibility
- **One-Line Integration:** Works as a transparent proxy. If your code works with OpenAI, it works with LEO Optima.
- **Model Agnostic:** Use it with GPT-4, Claude 3, Llama 3, or any custom endpoint.

---

## 📊 Analytics & Monitoring
LEO Optima now includes a built-in analytics engine to track your savings in real-time.

### Accessing Analytics
You can retrieve live statistics by sending a GET request to the `/v1/analytics` endpoint:
```bash
curl http://localhost:8000/v1/analytics
```

**Example Response:**
```json
{
  "total_queries": 150,
  "total_cost_saved": 12.45,
  "total_cost_spent": 1.20,
  "cache_hits": 85,
  "history": [...]
}
```

## 🚀 How It Works: The "Smart Bridge"

1. **The Request:** Your application sends a request to LEO Optima instead of the direct API provider.
2. **The Check:** LEO's **Novelty & Coherence Engines** analyze the query.
3. **The Decision:**
   - **Cache Hit:** If a similar query exists, LEO returns the answer immediately for **$0 cost**.
   - **Cache Miss:** If the query is new, LEO forwards it to your provider, retrieves the answer, and stores it for future use.

---

## 🛠 Quick Start

### 1. Installation
```bash
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima
pip install -r requirements.txt # (Coming soon) or install: fastapi uvicorn httpx numpy
```

### 2. Run the Proxy
```bash
python3 proxy_server.py
```

### 3. Connect Your App
Just change your `base_url` to point to LEO Optima:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your_api_key",
    base_url="http://localhost:8000/v1" # This is the magic line
)
```

---

## 📊 Business Impact
| Metric | Without LEO Optima | With LEO Optima |
| :--- | :--- | :--- |
| **Cost per Duplicate Query** | Full Price ($$$) | **$0.00** |
| **Response Time** | 2,000ms - 10,000ms | **<10ms** |
| **API Reliability** | Subject to Provider Downtime | **Always Available** (for cached queries) |
| **Data Privacy** | Sent to Cloud Providers | **Stored Locally** |

---

## 🗺 Roadmap
For a detailed view of our progress and upcoming features, please refer to the [ROADMAP.md](./ROADMAP.md).

- [x] **v1.0 - v1.2:** Core Proxy, Semantic Cache, and Analytics Backend.
- [ ] **v1.3:** Multi-tenant Support & API Key Management.
- [ ] **v1.4:** Analytics UI Dashboard.

---

## 📄 License
LEO Optima is a proprietary optimization engine. For licensing inquiries or enterprise support, please visit [leo-optima.com](https://leo-optima.com).

---
*Built with ❤️ for the AI Developer Community.*
