# LEO Optima Community Quick Start

Get LEO Optima up and running on your local machine in less than 5 minutes.

---

## 🚀 1. Setup (2 minutes)

### Docker (Recommended)
```bash
git clone https://github.com/BADJAB22/leo-optima.git
cd leo-optima
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
docker compose up --build -d
```

### Manual
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export LEO_API_KEY="any_secret_key"
python proxy_server.py
```

---

## 🔌 2. Connect (1 minute)

Update your OpenAI client to point to your local LEO Optima instance.

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...", # Your OpenAI Key
    base_url="http://localhost:8000/v1",
    default_headers={"X-API-Key": "your_leo_key"}
)
```

---

## 📊 3. Monitor (1 minute)

Open your browser and go to:
**http://localhost:3000**

Login with the `LEO_API_KEY` you set in your `.env` file. You can now see your savings and optimization metrics in real-time.

---

## 💡 Pro Tip
LEO Optima works best when you use the same model consistently, as it allows the semantic cache to build up a rich history of your specific use cases.

Happy Optimizing! 🦁
