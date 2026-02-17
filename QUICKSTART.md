# ⚡ LEO Optima Quick Start Guide

Welcome to the future of cost-efficient AI! This guide will get you from "Zero" to "Saving Money" in less than 2 minutes.

---

## 🛠️ Step 1: Environment Setup

LEO Optima is designed to be plug-and-play. Choose your preferred way to run:

### A. The "I want it now" way (Docker)
This is the most stable way. It sets up the API, the Dashboard, and the Redis cache in one go.

```bash
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima
cp .env.example .env
# Open .env and add your API keys (OpenAI, Anthropic, etc.)
docker compose up --build -d
```

### B. The "I'm a Python purist" way (Manual)
```bash
# Install the essentials
pip install -r requirements.txt

# Set your secrets
export OPENAI_API_KEY="sk-..."
export LEO_API_KEY="create_a_strong_password_here"

# Start the engine
python proxy_server.py
```

---

## 🔌 Step 2: Connect Your App

You don't need to rewrite your code. Just point your existing OpenAI client to LEO.

### Python
```python
from openai import OpenAI

client = OpenAI(
    api_key="your_actual_openai_key",
    base_url="http://localhost:8000/v1", # The magic line
    default_headers={"X-API-Key": "your_leo_key"}
)
```

### Node.js
```javascript
const OpenAI = require("openai");

const client = new OpenAI({
  apiKey: "your_actual_openai_key",
  baseURL: "http://localhost:8000/v1",
  defaultHeaders: { "X-API-Key": "your_leo_key" }
});
```

---

## 📊 Step 3: Watch the Savings

Once you've sent a few requests, head over to your browser:

👉 **[http://localhost:3000](http://localhost:3000)**

Login with your `LEO_API_KEY`. You'll see:
- **Total USD Saved**: Watch your balance stay in your pocket.
- **Cache Hits**: See how many requests LEO answered without hitting the paid API.
- **Token Efficiency**: A real-time view of your prompt optimization.

---

## 💡 Pro Tips for Maximum ROI

1.  **Consistency is King**: The more you use the same model (e.g., `gpt-4o`), the better LEO gets at caching and optimizing for that specific logic.
2.  **Feedback Loop**: Use the dashboard to see which queries are hitting the cache. If a query should have been cached but wasn't, you can adjust your similarity thresholds in the config.
3.  **Multi-LLM**: Don't forget that LEO can route to different providers. Try mixing Claude for complex tasks and GPT for fast ones!

---

**Need help?** Reach out to [**@BADJAB22**](https://twitter.com/BADJAB22) on Twitter or open an issue on GitHub. 

Let's build smarter, not more expensive! 🦁
