# 🔌 LEO Optima: The Enterprise Integration Guide

This guide provides everything a developer needs to integrate **LEO Optima** into a production environment, ensuring maximum cost savings and reliability.

---

## 🎯 The Philosophy: "Intelligence as a Layer"

At **Kadropic Labs**, we believe that LLM orchestration should be invisible. LEO Optima is designed as a "Smart Financial Layer"—it handles the complexity of caching, routing, and verification so you can focus on building features.

---

## 🛠️ Supported LLM Providers

LEO Optima is a unified gateway. While it uses OpenAI-compatible endpoints, it can be extended to support any provider:

- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-Turbo.
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus.
- **Google**: Gemini 1.5 Pro/Flash.
- **Local Models**: Support for Ollama, vLLM, and local inference engines.

*To add a custom provider, implement the `LLMInterface` in `api_interfaces.py`.*

---

## 🐍 Python SDK Integration

```python
import os
from openai import OpenAI

# 1. Configure the LEO Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="http://localhost:8000/v1", # LEO Optima Proxy
    default_headers={
        "X-API-Key": os.getenv("LEO_API_KEY") # Your local LEO security key
    }
)

# 2. Execute with Confidence
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Generate a technical audit for our AI infrastructure."}]
)

# 3. Access Optimization Metadata
if hasattr(response, 'leo_metrics'):
    print(f"Optimization Route: {response.leo_metrics['route']}")
    print(f"Confidence Score: {response.leo_metrics['confidence'] * 100}%")
```

---

## 📊 Analytics & Monitoring

The **Kadropic Dashboard** (available at `http://localhost:3000`) provides a high-level overview of your AI economy:

- **Cost Savings (USD)**: Real-time calculation based on tokens saved vs. provider market rates.
- **Semantic Hit Rate**: The percentage of queries served from the local intelligent cache.
- **Decomposition Ratio**: Percentage of complex queries broken down for efficiency.

---

## 🛡️ Production Best Practices

1.  **Security**: Always set a strong `LEO_API_KEY` in your environment to prevent unauthorized access to your local proxy.
2.  **Storage**: LEO stores its intelligence (vector cache and identity DB) in `leo_storage/`. Ensure this directory is persistent.
3.  **Redis Scaling**: For high-concurrency environments, ensure the Redis service has sufficient memory for the semantic lookup table.

---

## 🚀 About Kadropic Labs

**Kadropic Labs** is a specialized AI research and development firm founded by **Bader Jamal**. We focus on building tools that bridge the gap between cutting-edge AI research and practical, cost-effective deployment.

- **Founder**: [Bader Jamal](https://www.linkedin.com/in/bader-jamal-466a6b86)
- **Official Site**: [kadropiclabs.com](https://kadropiclabs.com)
- **Twitter**: [@baderjamal0](https://twitter.com/baderjamal0)

---
**License**: MIT. Built with precision for the global developer community.
