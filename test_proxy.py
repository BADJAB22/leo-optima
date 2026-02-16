import requests
import time

def test_proxy():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-dummy-key", # Not needed for local simulation if we mock the backend
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
    }

    print("--- Test 1: First request (Should be forwarded/failed if no key) ---")
    try:
        start = time.time()
        # Since we don't have a real key, this might fail, but let's see the proxy logic
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:100]}...")
        print(f"Time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Error: {e}")

    # Let's manually inject something into the cache to test optimization
    print("\n--- Manually Injecting into Cache for Demo ---")
    from core_engine import LeoOptimizer
    opt = LeoOptimizer()
    opt.update_cache("What is the capital of France?", "The capital of France is Paris.")
    
    print("\n--- Test 2: Second request (Should hit LEO Cache) ---")
    start = time.time()
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()['choices'][0]['message']['content']}")
    print(f"LEO Metrics: {response.json().get('leo_metrics')}")
    print(f"Time: {time.time() - start:.4f}s (SUPER FAST & FREE)")

if __name__ == "__main__":
    test_proxy()
