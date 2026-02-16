from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import json
import os
import time
from Truth_Optima import TruthOptima

app = FastAPI(title="LEO Optima Universal Proxy")

# Initialize TruthOptima for advanced routing and analytics
# We no longer use LeoOptimizer from core_engine.py as TruthOptima is the unified engine.
truth_system = TruthOptima()

@app.get("/v1/analytics")
async def get_analytics():
    """Return the current LEO Optima analytics and savings data"""
    return truth_system.stats

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    # Extract the last user message
    last_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found in 'messages'")
    
    if stream:
        return StreamingResponse(
            stream_generator(last_user_message, body.get("model", "optimized-model")),
            media_type="text/event-stream"
        )

    # Use TruthOptima for intelligent routing and analytics tracking
    response_obj = await truth_system.ask(last_user_message)
    
    # Format response in OpenAI-compatible format
    response_data = {
        "id": f"leo-opt-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "optimized-model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_obj.answer},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "leo_metrics": {
            "route": response_obj.route.value,
            "confidence": round(response_obj.confidence, 4),
            "risk_level": response_obj.risk_level.value,
            "cost_estimate": response_obj.cost_estimate,
            "novelty": round(response_obj.novelty, 4) if response_obj.novelty else None,
            "coherence": round(response_obj.coherence, 4) if response_obj.coherence else None,
        }
    }
    
    if response_obj.proof:
        response_data["leo_metrics"]["proof"] = {
            "valid": response_obj.proof.is_valid(truth_system.config.tau_sigma),
            "commitment": response_obj.proof.commitment,
            "sigma": round(response_obj.proof.sigma, 4)
        }

    return response_data

async def stream_generator(question: str, model: str):
    """Generator for OpenAI-compatible SSE streaming"""
    request_id = f"leo-opt-stream-{int(time.time())}"
    
    async for chunk in truth_system.stream(question):
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
    
    # Final chunk
    final_data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
