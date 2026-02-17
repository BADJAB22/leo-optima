# 🦁 LEO Optima: The Intelligent LLM Cost-Reduction Layer

<div align="center">
  <img src="https://img.shields.io/badge/Maintained%20by-Kadropic%20Labs-blue?style=for-the-badge" alt="Kadropic Labs">
  <img src="https://img.shields.io/badge/Author-Bader%20Jamal-orange?style=for-the-badge" alt="Bader Jamal">
  <br>
  <a href="https://kadropiclabs.com"><img src="https://img.shields.io/badge/Website-kadropiclabs.com-lightgrey?style=flat-square&logo=google-chrome" alt="Website"></a>
  <a href="https://twitter.com/baderjamal0"><img src="https://img.shields.io/badge/X-@baderjamal0-black?style=flat-square&logo=x" alt="X"></a>
  <a href="https://www.linkedin.com/in/bader-jamal-466a6b86"><img src="https://img.shields.io/badge/LinkedIn-Bader%20Jamal-blue?style=flat-square&logo=linkedin" alt="LinkedIn"></a>
</div>

---

## 🚀 Stop Burning Tokens. Start Building Intelligence.

**LEO Optima** is a high-performance, self-hosted orchestration engine that slashes LLM API costs by **60-80%** while simultaneously improving response quality. Designed for developers and enterprises who demand efficiency without compromise.

Built by **[Bader Jamal](https://github.com/BADJAB22)** at **[Kadropic Labs](https://kadropiclabs.com)**.

---

## 💡 Why LEO Optima?

In the current AI landscape, tokens are the new gold. Most applications bleed money through redundant queries, bloated prompts, and over-provisioned models. LEO Optima acts as a **Smart Financial Layer** for your AI stack.

### 🦞 The OpenClaw Solution
Are you running **OpenClaw** (formerly Moltbot)? Autonomous agents like OpenClaw are incredible but notorious for "burning" tokens through recursive loops and repetitive state-checks. 

**LEO Optima is the perfect companion for your OpenClaw installation.**
- **Loop Deduplication**: Prevents paying for the same state-check queries during agent loops.
- **Context Slimming**: Slashes the cost of long-running agent conversations by stripping redundant system prompts.
- **Cost Guardrails**: Monitor exactly how much your OpenClaw agent is spending in real-time.

> **Note**: For the ultimate autonomous experience, **Install LEO Optima alongside your OpenClaw setup** and point your OpenClaw `OPENAI_BASE_URL` to LEO.

### 💰 The Value Proposition
- **Drastic Cost Reduction**: Automatically saves up to 80% on OpenAI, Anthropic, and Gemini bills.
- **Sub-Millisecond Speed**: Serve repeated or semantically similar queries instantly from your local cache.
- **Provider Agnostic**: One unified interface for GPT-5, Claude 4.5, Gemini, and local models.
- **Verifiable Truth**: Built-in Byzantine Consensus ensures you get the most accurate answer every time.

---

## 🧠 The Optimization Core

LEO Optima isn't just a proxy; it's a multi-stage intelligence pipeline:

1.  **Request Deduplication**: Identical concurrent requests are batched. Process once, serve all—zero extra cost.
2.  **Adaptive Semantic Cache**: Powered by **Johnson-Lindenstrauss Projection**, LEO understands the *intent* of your queries, serving cached answers even if the wording changes.
3.  **Query Decomposition**: Complex tasks are broken into atomic, cacheable fragments to maximize future reuse.
4.  **Prompt Slimming**: Automatically removes "token fluff" from your prompts before they hit the paid API.
5.  **Byzantine Verification**: Cross-verifies high-stakes queries across multiple models for absolute reliability.

---

## 📊 Real-Time Savings Dashboard

Don't just take our word for it. Monitor your ROI in real-time with the built-in **Kadropic Analytics Dashboard**:
- **Live USD Savings**: Track every cent saved from hitting the paid APIs.
- **Cache Performance**: Visual hit-rate metrics and optimization ratios.
- **Route Intelligence**: See how LEO routes your traffic between Cache, Fast, and Consensus paths.

---

## 🛠️ Quick Start (120 Seconds)

### Deploy with Docker (Recommended)
```bash
# Clone the core engine
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima

# Configure your secrets
cp .env.example .env
# Add your Provider Keys (OpenAI, Anthropic, etc.) and your secret LEO_API_KEY

# Ignite
docker compose up --build -d
```

- **Dashboard**: `http://localhost:3000`
- **API Proxy**: `http://localhost:8000`

### 🔌 Using with OpenClaw
Simply update your OpenClaw `.env` or environment variables:
```bash
OPENAI_BASE_URL=http://localhost:8000/v1
# Ensure your LEO_API_KEY is passed in headers if required by your setup
```

---

## 🔌 Drop-in Integration

LEO Optima is a **seamless replacement** for your existing OpenAI SDK setup. Change one line of code, and you're saving money.

```python
from openai import OpenAI

# Simply point to your local LEO instance
client = OpenAI(
    api_key="your-provider-key", 
    base_url="http://localhost:8000/v1",
    default_headers={"X-API-Key": "your_secret_leo_key"}
)

# Use as normal—LEO handles the optimization magic
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze our Q4 growth strategy."}]
)
```

---

## 🤝 Community & Support

LEO Optima is **100% Free and Open Source**. We believe high-performance AI should be accessible to every visionary developer.

- **Found a bug?** Open an [Issue](https://github.com/BADJAB22/leo-optima/issues).
- **Want to contribute?** We love [Pull Requests](https://github.com/BADJAB22/leo-optima/pulls).
- **Direct Contact?** Connect with me on [**LinkedIn**](https://www.linkedin.com/in/bader-jamal-466a6b86) or [**X**](https://twitter.com/baderjamal0).

---
**Crafted with precision by [Kadropic Labs](https://kadropiclabs.com).**
**License**: MIT. Built for the builders.
