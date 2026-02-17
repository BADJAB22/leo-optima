# 🦁 LEO Optima: The Intelligent LLM Cost-Reduction Layer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![Twitter Follow](https://img.shields.io/twitter/follow/BADJAB22?style=social)](https://twitter.com/BADJAB22)

**LEO Optima** is a high-performance, self-hosted orchestration engine designed to slash your LLM API costs by **60-80%** without sacrificing a shred of quality. It sits intelligently between your application and your AI providers, ensuring every token spent is a token earned.

Built by **[BADJAB](https://github.com/BADJAB22)** | [badjab.io](https://badjab.io)

---

## 🚀 Why LEO Optima?

In the world of Generative AI, tokens are currency. Most applications waste thousands of dollars on redundant queries, over-complicated prompts, and expensive models for simple tasks. 

LEO Optima changes the game by treating your AI calls with the same efficiency as a high-frequency trading system.

### 💡 Key Benefits
- **💰 Massive Savings**: Reduce OpenAI, Anthropic, and Gemini bills by up to 80% through aggressive semantic caching and prompt optimization.
- **⚡ Blazing Speed**: Serve repeated or similar queries in milliseconds directly from your local Redis cache.
- **🛡️ Quality Assurance**: Built-in Byzantine Consensus and Confidence Scoring ensure you never get a "hallucinated" or low-quality response.
- **🌐 Multi-LLM Native**: One interface to rule them all. Switch between GPT-4, Claude 3.5, and Gemini without changing a line of code.

---

## 🧠 The Optimization Engine

LEO Optima isn't just a proxy; it's a brain. It uses a sophisticated pipeline to handle every request:

1.  **Request Deduplication**: Identical requests waiting in line? LEO processes one and serves all, for free.
2.  **Adaptive Semantic Cache**: Using **Johnson-Lindenstrauss Projection**, LEO understands the *meaning* of your queries, serving similar answers even if the wording differs.
3.  **Query Decomposition**: Complex questions are broken down into smaller, cacheable fragments, maximizing reuse.
4.  **Prompt Slimming**: Automatically strips redundant tokens and "fluff" from your prompts before they hit the paid API.
5.  **Byzantine Verification**: For high-risk queries, LEO can cross-verify across multiple models to ensure absolute truth.

---

## 📊 Professional Dashboard

Stop guessing your savings. LEO Optima comes with a sleek, real-time dashboard that gives you the full picture:
- **Live Cost Tracking**: See exactly how much you've saved in USD.
- **Token Efficiency**: Track your optimization ratio and cache hit rates.
- **Route Analysis**: Visualize how LEO routes your traffic between Cache, Fast, and Consensus paths.

---

## 🛠️ Quick Start (Get running in 120 seconds)

### The Docker Way (Recommended)
```bash
# Clone the masterpiece
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima

# Set your keys
cp .env.example .env
# Open .env and add your OPENAI_API_KEY and a secret LEO_API_KEY

# Ignite the engine
docker compose up --build -d
```

- **Dashboard**: `http://localhost:3000`
- **API Proxy**: `http://localhost:8000`

---

## 🔌 One-Line Integration

LEO Optima is a **drop-in replacement**. If your code works with OpenAI, it works with LEO.

```python
from openai import OpenAI

# Just point the base_url to your local LEO instance
client = OpenAI(
    api_key="your-provider-key", 
    base_url="http://localhost:8000/v1",
    default_headers={"X-API-Key": "your_secret_leo_key"}
)

# LEO handles the rest - saving you money on every call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing simply."}]
)
```

---

## 🤝 Join the Movement

This is a **100% Free and Open Source** project. We believe high-performance AI should be affordable for every developer.

- **Found a bug?** Open an [Issue](https://github.com/BADJAB22/leo-optima/issues).
- **Have an idea?** Submit a [Pull Request](https://github.com/BADJAB22/leo-optima/pulls).
- **Want to chat?** Follow me on Twitter [**@BADJAB22**](https://twitter.com/BADJAB22).

Built with ❤️ by **BADJAB** for the global AI community.

---
**License**: MIT. Free to use, modify, and scale.
