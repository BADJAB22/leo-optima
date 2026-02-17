# 🦁 LEO Optima: Open Source LLM Optimization Layer

**LEO Optima Community Edition** is a high-performance, self-hosted routing and optimization engine designed to reduce LLM API costs by **60-80%** while maintaining response quality. 

This project is **100% Free and Open Source**, built for the community to make advanced LLM orchestration accessible to everyone.

---

## 🚀 Key Features

### 📊 Community Dashboard
LEO Optima includes a built-in web dashboard for real-time monitoring of your local optimization engine:
- **Real-time Metrics**: Monitor cost savings, token optimization, and cache performance.
- **Usage Tracking**: Keep an eye on your token usage and cost estimates.
- **Performance Visualizations**: Interactive charts showing optimization trends.

### 🧠 Optimization Pipeline
1.  **Request Deduplication**: Automatically batches identical requests to prevent redundant API calls.
2.  **Adaptive Semantic Cache**: Uses dynamic similarity thresholds to serve similar queries from local storage (Redis + SQLite).
3.  **Query Decomposition**: Breaks complex queries into simpler parts to maximize cache reuse.
4.  **Prompt Optimization**: Strips redundant tokens from prompts to save costs directly.
5.  **Confidence Scoring**: Scores model responses and automatically retries if quality is below threshold.

---

## 🛠️ Quick Start

### 1. Run with Docker (Recommended)
The easiest way to get started is using Docker Compose, which sets up the API, the Dashboard, and a Redis cache automatically.

```bash
# Clone the repository
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima

# Setup environment
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY and a secret LEO_API_KEY

# Start everything
docker compose up --build -d
```

- **API Server**: `http://localhost:8000`
- **Dashboard**: `http://localhost:3000`

### 2. Manual Installation
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export LEO_API_KEY="your_secret_key"
python proxy_server.py
```

---

## 🔌 Integration

LEO Optima is a **drop-in replacement** for OpenAI. Simply change your `base_url` and add the `X-API-Key` header.

### Python Example
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-openai-key",
    base_url="http://localhost:8000/v1",
    default_headers={"X-API-Key": "your_leo_key"}
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## 🤝 Contributing
We welcome contributions from the community! Whether it's a bug fix, a new feature, or improved documentation, feel free to open an issue or submit a pull request.

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
**Built with ❤️ for the AI Community.**
