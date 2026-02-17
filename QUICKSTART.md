# ⚡ LEO Optima: 120-Second Deployment Guide

Welcome to **LEO Optima**, the industry's most efficient LLM cost-reduction layer. Let's get you set up to start saving money immediately.

---

## 🏗️ 1. Environment Setup

LEO Optima is built for stability and ease of use. Choose your deployment path:

### Option A: Docker (Recommended for Production)
This path deploys the API, the Analytics Dashboard, and the Redis cache layer in a single command.

```bash
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima
cp .env.example .env
# Open .env and add your API keys (OpenAI, Claude, etc.)
docker compose up --build -d
```

### Option B: Manual (Development Only)
```bash
# Install dependencies
pip install -r requirements.txt

# Set your environment
export OPENAI_API_KEY="sk-..."
export LEO_API_KEY="your_custom_secure_key"

# Run the engine
python proxy_server.py
```

---

## 🔌 2. Seamless Integration

Update your existing application to route through LEO Optima. No logic changes required.

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    api_key="your_actual_api_key",
    base_url="http://localhost:8000/v1", # Route through LEO
    default_headers={"X-API-Key": "your_leo_key"}
)
```

### Node.js (OpenAI SDK)
```javascript
const OpenAI = require("openai");

const client = new OpenAI({
  apiKey: "your_actual_api_key",
  baseURL: "http://localhost:8000/v1",
  defaultHeaders: { "X-API-Key": "your_leo_key" }
});
```

---

## 📊 3. Access the Dashboard

Monitor your savings and performance in real-time:

👉 **[http://localhost:3000](http://localhost:3000)**

Login using the `LEO_API_KEY` defined in your environment. You'll get instant visibility into:
- **Total USD Saved**
- **Token Optimization Ratios**
- **Semantic Cache Hit Rates**

---

## 🦁 About the Author

LEO Optima is a project by **Bader Jamal**, founder of **Kadropic Labs**. We are dedicated to building tools that make the AI revolution affordable and sustainable.

- **Website**: [kadropiclabs.com](https://kadropiclabs.com)
- **LinkedIn**: [Bader Jamal](https://www.linkedin.com/in/bader-jamal-466a6b86)
- **X (Twitter)**: [@baderjamal0](https://twitter.com/baderjamal0)

---
**Happy Optimizing!**
