from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import json
import os
import time
from datetime import datetime
from typing import Dict
from Truth_Optima import TruthOptima
from leo_optima_single_model import (
    LEOOptimaSingleModel,
    PromptOptimizer,
    ConfidenceScorer,
)

app = FastAPI(title="LEO Optima Universal Proxy v1.0")

# Initialize TruthOptima for advanced routing
truth_system = TruthOptima()

# Initialize single-model optimizer with 5 strategies
single_model_optimizer = LEOOptimaSingleModel(use_local_fallback=False)

# Optimization configuration
OPTIMIZATION_CONFIG = {
    'adaptive_cache': True,
    'query_decomposition': True,
    'prompt_optimization': True,
    'confidence_scoring': True,
    'request_deduplication': True,
}

# Metrics tracking
class OptimizationMetrics:
    """Track optimization performance"""
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'decomposed_queries': 0,
            'low_confidence_retries': 0,
            'tokens_saved': 0,
            'cost_saved': 0.0,
            'average_confidence': 0.0,
        }
        self.request_log = []
    
    def record_request(self, cache_hit=False, decomposed=False, confidence=1.0, tokens_saved=0, cost_saved=0.0):
        """Record a request with all metrics"""
        self.metrics['total_requests'] += 1
        
        if cache_hit:
            self.metrics['cache_hits'] += 1
        
        if decomposed:
            self.metrics['decomposed_queries'] += 1
        
        self.metrics['tokens_saved'] += tokens_saved
        self.metrics['cost_saved'] += cost_saved
        
        # Update average confidence
        total_conf = self.metrics['average_confidence'] * (self.metrics['total_requests'] - 1)
        self.metrics['average_confidence'] = (total_conf + confidence) / self.metrics['total_requests']
        
        self.request_log.append({
            'timestamp': datetime.now().isoformat(),
            'cache_hit': cache_hit,
            'decomposed': decomposed,
            'confidence': confidence,
            'tokens_saved': tokens_saved,
        })
        
        if len(self.request_log) > 1000:
            self.request_log = self.request_log[-1000:]
    
    def get_summary(self):
        """Get metrics summary"""
        return {
            **self.metrics,
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['total_requests']),
            'average_tokens_saved_per_request': self.metrics['tokens_saved'] / max(1, self.metrics['total_requests']),
        }

metrics = OptimizationMetrics()

# ============================================================
# ORIGINAL ENDPOINTS (Backward Compatible)
# ============================================================

@app.get("/v1/analytics")
async def get_analytics():
    """Return comprehensive analytics including optimization metrics"""
    return {
        'leo_optima_stats': truth_system.stats,
        'optimization_metrics': metrics.get_summary(),
        'single_model_stats': single_model_optimizer.get_stats(),
        'timestamp': datetime.now().isoformat()
    }

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

    # Process with optimizations
    start_time = time.time()
    
    # 1. Check for decomposition
    decomposed = False
    if OPTIMIZATION_CONFIG['query_decomposition']:
        if single_model_optimizer.decomposer.should_decompose(last_user_message):
            decomposed = True
    
    # 2. Optimize prompt
    optimized_query = last_user_message
    if OPTIMIZATION_CONFIG['prompt_optimization']:
        optimized_query = single_model_optimizer.prompt_optimizer.optimize_prompt(last_user_message)
    
    # 3. Use main system for response
    response_obj = await truth_system.ask(optimized_query)
    
    # 4. Score confidence
    confidence = 1.0
    if OPTIMIZATION_CONFIG['confidence_scoring']:
        confidence = single_model_optimizer.confidence_scorer.score_response(
            last_user_message, 
            response_obj.answer
        )
    
    # 5. Retry if confidence is low
    if confidence < 0.6 and OPTIMIZATION_CONFIG['confidence_scoring']:
        metrics.metrics['low_confidence_retries'] += 1
        rephrased = single_model_optimizer._rephrase_query(last_user_message)
        response_obj = await truth_system.ask(rephrased)
        confidence = single_model_optimizer.confidence_scorer.score_response(
            last_user_message,
            response_obj.answer
        )
    
    # Calculate tokens saved (rough estimate)
    original_tokens = len(last_user_message.split()) * 1.3
    optimized_tokens = len(optimized_query.split()) * 1.3
    tokens_saved = max(0, original_tokens - optimized_tokens)
    
    # Record metrics
    cache_hit = response_obj.route.value == "CACHE"
    metrics.record_request(
        cache_hit=cache_hit,
        decomposed=decomposed,
        confidence=confidence,
        tokens_saved=int(tokens_saved),
        cost_saved=response_obj.cost_estimate
    )
    
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
        "usage": {
            "prompt_tokens": int(original_tokens),
            "completion_tokens": int(optimized_tokens),
            "total_tokens": int(original_tokens + optimized_tokens)
        },
        "leo_metrics": {
            "route": response_obj.route.value,
            "confidence": round(response_obj.confidence, 4),
            "risk_level": response_obj.risk_level.value,
            "cost_estimate": response_obj.cost_estimate,
            "novelty": round(response_obj.novelty, 4) if response_obj.novelty else None,
            "coherence": round(response_obj.coherence, 4) if response_obj.coherence else None,
            "verification_id": response_obj.verification_id,
            "audit_log": response_obj.audit_log
        },
        "optimization_metrics": {
            "cache_hit": cache_hit,
            "decomposed": decomposed,
            "confidence_score": round(confidence, 4),
            "tokens_saved": int(tokens_saved),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
    }
    
    if response_obj.proof:
        response_data["leo_metrics"]["proof"] = {
            "valid": response_obj.proof.is_valid(truth_system.config.tau_sigma),
            "commitment": response_obj.proof.commitment,
            "sigma": round(response_obj.proof.sigma, 4)
        }

    # Add Dynamic Verification Headers
    headers = {
        "X-LEO-Verification-ID": response_obj.verification_id,
        "X-LEO-Route": response_obj.route.value,
        "X-LEO-Confidence": str(round(response_obj.confidence, 4)),
        "X-LEO-Risk": response_obj.risk_level.value,
        "X-LEO-Proof-Valid": str(response_obj.proof.is_valid(truth_system.config.tau_sigma)) if response_obj.proof else "N/A"
    }

    return JSONResponse(content=response_data, headers=headers)

async def stream_generator(question: str, model: str):
    """Generator for OpenAI-compatible SSE streaming with optimizations"""
    request_id = f"leo-opt-stream-{int(time.time())}"
    
    # Optimize prompt if enabled
    optimized_question = question
    if OPTIMIZATION_CONFIG['prompt_optimization']:
        optimizer = PromptOptimizer()
        optimized_question = optimizer.optimize_prompt(question)
    
    async for chunk in truth_system.stream(optimized_question):
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

# ============================================================
# NEW OPTIMIZATION ENDPOINTS
# ============================================================

@app.get("/v1/optimization/status")
async def get_optimization_status():
    """Get current optimization configuration and status"""
    return {
        'config': OPTIMIZATION_CONFIG,
        'cache_size': len(single_model_optimizer.adaptive_cache.cache_entries),
        'cache_success_rate': single_model_optimizer.adaptive_cache.success_rate,
        'metrics': metrics.get_summary()
    }

@app.post("/v1/optimization/enable")
async def enable_optimization(request: Request):
    """Enable specific optimization strategy"""
    body = await request.json()
    strategy = body.get('strategy')
    enabled = body.get('enabled', True)
    
    if strategy not in OPTIMIZATION_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
    
    OPTIMIZATION_CONFIG[strategy] = enabled
    
    return {
        'strategy': strategy,
        'enabled': enabled,
        'config': OPTIMIZATION_CONFIG
    }

@app.post("/v1/optimization/cache/feedback")
async def provide_cache_feedback(request: Request):
    """Provide feedback on cache hit quality"""
    body = await request.json()
    cached_answer = body.get('cached_answer')
    is_correct = body.get('is_correct', True)
    
    single_model_optimizer.adaptive_cache.add_feedback(cached_answer, is_correct)
    
    return {
        'status': 'feedback_recorded',
        'cache_success_rate': single_model_optimizer.adaptive_cache.success_rate,
        'dynamic_threshold': single_model_optimizer.adaptive_cache.dynamic_threshold
    }

@app.get("/v1/optimization/cache/stats")
async def get_cache_stats():
    """Get detailed cache statistics"""
    cache = single_model_optimizer.adaptive_cache
    return {
        'total_entries': len(cache.cache_entries),
        'base_threshold': cache.base_threshold,
        'dynamic_threshold': cache.dynamic_threshold,
        'success_rate': cache.success_rate,
        'feedback_count': len(cache.feedback_history),
        'entries': [
            {
                'answer_preview': entry['answer'][:100],
                'confidence': entry['confidence'],
                'timestamp': entry['timestamp'].isoformat()
            }
            for entry in cache.cache_entries[-10:]
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
        "optimizations_active": sum(1 for v in OPTIMIZATION_CONFIG.values() if v)
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("LEO Optima v2.0 - Universal Proxy")
    print("=" * 60)
    print("✓ Adaptive Threshold Cache")
    print("✓ Query Decomposition")
    print("✓ Prompt Optimization")
    print("✓ Confidence Scoring")
    print("✓ Request Deduplication")
    print("=" * 60)
    print("Starting server on http://0.0.0.0:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
