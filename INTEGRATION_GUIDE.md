# 🔌 LEO Optima: The Developer's Integration Bible

So you've got LEO Optima running—congrats! Now let's talk about how to actually hook it up to your production apps and start saving those precious tokens.

---

## 🎯 The Philosophy: "Drop-in & Forget"

LEO Optima was built on one core principle: **Integration shouldn't be a headache.** 

Because LEO is fully OpenAI-compatible, it acts as a "smart proxy." Your code thinks it's talking to OpenAI, but it's actually talking to a genius middleman that knows how to save you money.

---

## 🛠️ Supported Providers

While the default is OpenAI, LEO's architecture is provider-agnostic. You can easily plug in:
- **OpenAI** (GPT-4o, GPT-4, GPT-3.5)
- **Anthropic** (Claude 3.5 Sonnet/Opus)
- **Google** (Gemini 1.5 Pro/Flash)
- **Local Models** (via Ollama or vLLM)

*Note: To add a new provider, check `api_interfaces.py` and implement the `LLMInterface`.*

---

## 🐍 Python Integration (The 10-Second Version)

```python
import os
from openai import OpenAI

# 1. Initialize with LEO as the base
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="http://localhost:8000/v1", # LEO's address
    default_headers={
        "X-API-Key": os.getenv("LEO_API_KEY") # Your local LEO security key
    }
)

# 2. Use it exactly like you normally would
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem about saving money."}]
)

# 3. (Optional) Check the magic metrics
# LEO injects extra info into the response object
if hasattr(response, 'leo_metrics'):
    print(f"Route taken: {response.leo_metrics['route']}")
    print(f"Confidence: {response.leo_metrics['confidence']}")
```

---

## 🌐 Dashboard & Monitoring

LEO isn't just a black box. It provides a professional-grade **Community Dashboard** at `http://localhost:3000`.

### What to look for:
- **Cost Saved**: Calculated based on current market rates for tokens.
- **Cache Hit Rate**: If this is above 30%, you're doing great. If it's below 10%, consider if your queries are too unique or if you need to lower the `gamma` (similarity threshold).
- **Optimization Split**: Shows you how many requests were saved by the Cache vs. how many needed the full API.

---

## 🛡️ Security & Production Best Practices

1.  **Protect your LEO_API_KEY**: This key is what secures your local LEO instance. Don't leak it!
2.  **Redis is your friend**: For production workloads, always run LEO with Redis (the default in our Docker setup). It makes the semantic search significantly faster.
3.  **Persistence**: LEO stores its "memory" and "identity" in the `leo_storage/` folder. Make sure this folder is backed up or mounted as a volume in Docker.

---

## 🚀 Advanced: Tuning the "Brain"

If you're a power user, you can tune LEO's behavior in `Truth_Optima.py` or via environment variables:

- **`LEO_CACHE_THRESHOLD`**: (Default 0.45) Lower this to get more cache hits (be careful with quality!).
- **`LEO_DEDUP_WINDOW`**: How many seconds to wait for a duplicate request before giving up.
- **`LEO_CONFIDENCE_MIN`**: The minimum score a response needs before LEO accepts it.

---

**Happy Coding!** 
Built with ❤️ by **[BADJAB](https://twitter.com/BADJAB22)**. If LEO saves you money, buy me a coffee or just give the repo a ⭐!
