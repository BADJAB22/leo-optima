# LEO Optima API Documentation v2.0

Complete reference for all API endpoints with examples and response formats.

---

## Base URL

```
http://localhost:8000
```

All endpoints are prefixed with `/v1` unless otherwise specified.

---

## Authentication

Currently, LEO Optima uses the same authentication as your upstream LLM provider (e.g., OpenAI API key). Future versions will support:
- API key authentication
- OAuth 2.0
- JWT tokens

---

## Core Endpoints

### 1. Chat Completions (OpenAI Compatible)

**Endpoint:** `POST /v1/chat/completions`

**Description:** Main endpoint for sending queries. Fully compatible with OpenAI's chat completions API.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": false
  }'
```

**Response:**
```json
{
  "id": "leo-opt-1708000000",
  "object": "chat.completion",
  "created": 1708000000,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  },
  "leo_metrics": {
    "route": "CONSENSUS",
    "confidence": 0.92,
    "risk_level": "LOW",
    "cost_estimate": 0.0025,
    "novelty": 0.45,
    "coherence": 0.38,
    "proof": {
      "valid": true,
      "commitment": "a1b2c3d4e5f6g7h8",
      "sigma": 0.89
    }
  },
  "optimization_metrics": {
    "cache_hit": false,
    "decomposed": false,
    "confidence_score": 0.92,
    "tokens_saved": 12,
    "processing_time_ms": 1245.5
  }
}
```

**Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model` | string | Yes | Model to use (e.g., "gpt-4", "gpt-3.5-turbo") |
| `messages` | array | Yes | Array of message objects |
| `temperature` | float | No | Sampling temperature (0-2) |
| `max_tokens` | integer | No | Maximum tokens in response |
| `stream` | boolean | No | Enable streaming responses |
| `top_p` | float | No | Nucleus sampling parameter |

**Response Fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `leo_metrics` | object | LEO Optima routing and verification metrics |
| `optimization_metrics` | object | Optimization strategy metrics |
| `choices[0].message.content` | string | The actual response |

---

### 2. Analytics Endpoint

**Endpoint:** `GET /v1/analytics`

**Description:** Get comprehensive analytics and performance metrics.

**Request:**
```bash
curl http://localhost:8000/v1/analytics
```

**Response:**
```json
{
  "leo_optima_stats": {
    "total_queries": 1250,
    "cache_hits": 450,
    "fast_routes": 600,
    "consensus_routes": 200,
    "total_cost": 12.50
  },
  "optimization_metrics": {
    "total_requests": 1250,
    "cache_hits": 450,
    "cache_hit_rate": 0.36,
    "decomposed_queries": 125,
    "low_confidence_retries": 35,
    "deduplicated_requests": 180,
    "tokens_saved": 12500,
    "cost_saved": 45.50,
    "average_confidence": 0.87,
    "average_tokens_saved_per_request": 10,
    "average_cost_per_request": 0.0364
  },
  "single_model_stats": {
    "total_queries": 1250,
    "cache_hits": 450,
    "decomposed_queries": 125,
    "local_model_used": 0,
    "total_tokens_saved": 12500,
    "total_cost_saved": 45.50,
    "dedup_stats": {
      "total_requests": 1250,
      "duplicates": 180,
      "dedup_rate": 0.144
    },
    "cache_size": 342,
    "cache_success_rate": 0.92
  },
  "timestamp": "2024-02-15T10:30:00.000Z"
}
```

---

### 3. Optimization Status

**Endpoint:** `GET /v1/optimization/status`

**Description:** Check which optimizations are active and their status.

**Request:**
```bash
curl http://localhost:8000/v1/optimization/status
```

**Response:**
```json
{
  "config": {
    "adaptive_cache": true,
    "query_decomposition": true,
    "prompt_optimization": true,
    "confidence_scoring": true,
    "request_deduplication": true
  },
  "cache_size": 342,
  "cache_success_rate": 0.92,
  "pending_requests": 3,
  "metrics": {
    "total_requests": 1250,
    "cache_hit_rate": 0.36,
    "average_confidence": 0.87,
    "cost_saved": 45.50
  }
}
```

---

### 4. Enable/Disable Optimization

**Endpoint:** `POST /v1/optimization/enable`

**Description:** Enable or disable specific optimization strategies.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/optimization/enable \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "query_decomposition",
    "enabled": false
  }'
```

**Response:**
```json
{
  "strategy": "query_decomposition",
  "enabled": false,
  "config": {
    "adaptive_cache": true,
    "query_decomposition": false,
    "prompt_optimization": true,
    "confidence_scoring": true,
    "request_deduplication": true
  }
}
```

**Valid Strategies:**
- `adaptive_cache`
- `query_decomposition`
- `prompt_optimization`
- `confidence_scoring`
- `request_deduplication`

---

### 5. Cache Feedback

**Endpoint:** `POST /v1/optimization/cache/feedback`

**Description:** Provide feedback on cache hit quality to improve adaptive threshold.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/optimization/cache/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "cached_answer": "Machine learning is a subset of AI...",
    "is_correct": true
  }'
```

**Response:**
```json
{
  "status": "feedback_recorded",
  "cache_success_rate": 0.92,
  "dynamic_threshold": 0.765
}
```

**Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `cached_answer` | string | Yes | The cached answer being evaluated |
| `is_correct` | boolean | Yes | Whether the answer was correct/helpful |

---

### 6. Cache Statistics

**Endpoint:** `GET /v1/optimization/cache/stats`

**Description:** Get detailed cache statistics and recent entries.

**Request:**
```bash
curl http://localhost:8000/v1/optimization/cache/stats
```

**Response:**
```json
{
  "total_entries": 342,
  "base_threshold": 0.9,
  "dynamic_threshold": 0.765,
  "success_rate": 0.92,
  "feedback_count": 125,
  "entries": [
    {
      "answer_preview": "Machine learning is a subset of artificial intelligence that focuses on...",
      "confidence": 0.95,
      "timestamp": "2024-02-15T10:25:00.000Z"
    },
    {
      "answer_preview": "Python is a high-level programming language known for its simplicity...",
      "confidence": 0.88,
      "timestamp": "2024-02-15T10:20:00.000Z"
    }
  ]
}
```

---

### 7. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the server is running and healthy.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-02-15T10:30:00.000Z",
  "version": "2.0-optimized",
  "optimizations_active": 5
}
```

---

## Streaming Responses

**Endpoint:** `POST /v1/chat/completions` (with `stream: true`)

**Description:** Get responses as a stream of Server-Sent Events (SSE).

**Request:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

**Response (SSE format):**
```
data: {"id":"leo-opt-1708000000","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"leo-opt-1708000000","object":"chat.completion.chunk","choices":[{"delta":{"content":" there"},"finish_reason":null}]}

data: {"id":"leo-opt-1708000000","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Code | Message | Solution |
| :--- | :--- | :--- |
| 400 | Invalid JSON body | Check request format |
| 400 | No user message found | Ensure messages array has user message |
| 400 | Unknown strategy | Use valid strategy name |
| 500 | Internal server error | Check server logs |

### Example Error Response

```json
{
  "detail": "Invalid JSON body"
}
```

---

## Rate Limiting

Currently, LEO Optima does not enforce rate limits, but your upstream provider might. Future versions will support:
- Per-user rate limits
- Per-IP rate limits
- Burst allowances

---

## Backward Compatibility

LEO Optima v2.0 is fully backward compatible with v1.0:
- All v1 endpoints still work
- New endpoints are additive
- Existing integrations require no changes

---

## Integration Examples

### Python (with requests library)

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def query_leo(message):
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": message}]
        }
    )
    return response.json()

def get_analytics():
    response = requests.get(f"{BASE_URL}/v1/analytics")
    return response.json()

# Usage
result = query_leo("What is LEO Optima?")
print(result['choices'][0]['message']['content'])

analytics = get_analytics()
print(f"Cache hit rate: {analytics['optimization_metrics']['cache_hit_rate']:.1%}")
```

### JavaScript (with fetch)

```javascript
const BASE_URL = "http://localhost:8000";

async function queryLEO(message) {
  const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gpt-4',
      messages: [{ role: 'user', content: message }]
    })
  });
  return await response.json();
}

async function getAnalytics() {
  const response = await fetch(`${BASE_URL}/v1/analytics`);
  return await response.json();
}

// Usage
const result = await queryLEO("What is LEO Optima?");
console.log(result.choices[0].message.content);

const analytics = await getAnalytics();
console.log(`Cache hit rate: ${(analytics.optimization_metrics.cache_hit_rate * 100).toFixed(1)}%`);
```

### cURL

```bash
# Query
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }' | jq '.choices[0].message.content'

# Analytics
curl http://localhost:8000/v1/analytics | jq '.optimization_metrics'

# Cache stats
curl http://localhost:8000/v1/optimization/cache/stats | jq '.cache_success_rate'
```

---

## Changelog

### v2.0 (Current)
- ✅ Adaptive Threshold Cache
- ✅ Query Decomposition
- ✅ Prompt Optimization
- ✅ Confidence Scoring
- ✅ Request Deduplication
- ✅ New optimization endpoints
- ✅ Enhanced analytics

### v1.0
- ✅ Byzantine Consensus
- ✅ Semantic Cache
- ✅ Novelty Detection
- ✅ Coherence Measurement

---

## Support

For issues or questions:
1. Check the [README](README_ENHANCED.md)
2. Review [Troubleshooting](README_ENHANCED.md#-troubleshooting)
3. Open a GitHub issue
4. Contact support@leo-optima.dev

---

**Last Updated:** February 15, 2024
**Version:** 2.0
