import requests
import time
import json

def test_analytics():
    base_url = "http://localhost:8000/v1"
    
    # 1. Check initial analytics
    print("--- Initial Analytics ---")
    resp = requests.get(f"{base_url}/analytics")
    print(json.dumps(resp.json(), indent=2))

    # 2. Send a few queries to trigger different routes
    queries = [
        "What is the capital of France?", # Likely FAST
        "What is the capital of France?", # Likely CACHE
        "Should I take 500mg of aspirin for a headache?", # Likely CONSENSUS (Medical)
        "How do I invest in stocks?" # Likely CONSENSUS (Financial)
    ]

    for q in queries:
        print(f"\n--- Sending Query: {q} ---")
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": q}]
        }
        resp = requests.post(f"{base_url}/chat/completions", json=payload)
        data = resp.json()
        print(f"Route: {data['leo_metrics'].get('route')}")
        print(f"Answer: {data['choices'][0]['message']['content'][:50]}...")

    # 3. Check final analytics
    print("\n--- Final Analytics ---")
    resp = requests.get(f"{base_url}/analytics")
    stats = resp.json()
    print(json.dumps(stats, indent=2))
    
    print(f"\nTotal Cost Saved: ${stats['total_cost_saved']:.4f}")
    print(f"Total Cost Spent: ${stats['total_cost_spent']:.4f}")

if __name__ == "__main__":
    # Start the server in the background would be needed here if not already running
    # For this test, we assume the user or another process starts it.
    test_analytics()
