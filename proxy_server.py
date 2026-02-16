from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import json
import os
from core_engine import LeoOptimizer
from Truth_Optima import TruthOptima

app = FastAPI(title="LEO Optima Universal Proxy")
optimizer = LeoOptimizer()
# Initialize TruthOptima for advanced routing and analytics
truth_system = TruthOptima()

@app.get("/v1/analytics")
async def get_analytics():
    """Return the current LEO Optima analytics and savings data"""
    return truth_system.stats

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    # Extract the last user message
    last_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # Use TruthOptima for intelligent routing and analytics tracking
    response_obj = await truth_system.ask(last_user_message)
    
    # If it's a cache hit, return immediately
    if response_obj.route.value == "CACHE":
        return {
            "id": f"leo-opt-{response_obj.timestamp}",
            "object": "chat.completion",
            "created": 123456789,
            "model": body.get("model", "optimized-model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_obj.answer},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "leo_metrics": {
                "route": response_obj.route.value,
                "confidence": response_obj.confidence,
                "cost_saved": 0.01
            }
        }

    # For non-cache, we'd normally forward to real API, 
    # but TruthOptima already handles the routing (FAST/CONSENSUS) 
    # and uses its internal models (simulated or real).
    
    return {
        "id": f"leo-opt-{response_obj.timestamp}",
        "object": "chat.completion",
        "created": 123456789,
        "model": body.get("model", "optimized-model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_obj.answer},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "leo_metrics": {
            "route": response_obj.route.value,
            "confidence": response_obj.confidence,
            "risk_level": response_obj.risk_level.value,
            "cost_estimate": response_obj.cost_estimate
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
