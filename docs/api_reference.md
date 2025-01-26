# ClaimCracker API Reference

## Base URL

```
https://claimcracker-api.onrender.com
```

## Rate Limiting

All endpoints are rate-limited to protect the API from abuse. Rate limits are specified per endpoint and are enforced using IP-based tracking.

Rate limit headers returned with each response:

- `X-RateLimit-Limit`: Maximum requests allowed in the time window
- `X-RateLimit-Remaining`: Number of requests remaining in the current window
- `X-RateLimit-Reset`: Time in seconds until the rate limit resets

## Authentication

Currently, the API is open for public use without authentication.

## Endpoints

### 1. Welcome Endpoint

```http
GET /
```

Returns API status and version information.

**Rate Limit:** 30 requests per minute

**Response Example:**

```json
{
  "message": "Welcome to ClaimCracker API",
  "version": "2.0.0",
  "status": "active"
}
```

### 2. Health Check

```http
GET /health
```

Checks if the API and model service are functioning correctly.

**Rate Limit:** 30 requests per minute

**Response Example:**

```json
{
  "status": "healthy"
}
```

### 3. Predict

```http
POST /predict
```

Analyzes text to determine if it contains fake news.

**Rate Limit:** 20 requests per minute

**Request Body:**

```json
{
  "text": "string (required)"
}
```

**Validation:**

- Text must not be empty
- Maximum length: 100,000 characters

**Response Example:**

```json
{
  "prediction": "Real", // or "Fake"
  "confidence": 0.95, // confidence score between 0 and 1
  "probabilities": {
    "Real": 0.95,
    "Fake": 0.05
  }
}
```

**Error Responses:**

- `422 Unprocessable Entity`: Invalid input
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Model prediction error

### 4. Cache Statistics

```http
GET /cache/stats
```

Returns statistics about the prediction cache.

**Rate Limit:** 10 requests per minute

**Response Example:**

```json
{
  "cache_hits": 150,
  "cache_misses": 50,
  "cache_size": 100,
  "max_size": 1000,
  "uptime": 3600.5
}
```

### 5. Clear Cache

```http
POST /cache/clear
```

Clears the prediction cache.

**Rate Limit:** 5 requests per minute

**Response Example:**

```json
{
  "status": "Cache cleared"
}
```

### 6. Metrics

```http
GET /metrics
```

Returns Prometheus metrics for monitoring.

**Rate Limit:** 30 requests per minute

**Available Metrics:**

- `model_prediction_latency_seconds`: Time spent processing model predictions
- `cache_hits_total`: Total number of cache hits
- `cache_misses_total`: Total number of cache misses
- `memory_usage_bytes`: Memory usage in bytes
- `cpu_usage_percent`: CPU usage percentage

**Response Format:** Prometheus text format

## CORS Support

The API supports Cross-Origin Resource Sharing (CORS) with the following configuration:

- Allowed Origins: `*` (all origins)
- Allowed Methods: All methods
- Allowed Headers: All headers
- Exposed Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common error status codes:

- `400`: Bad Request
- `422`: Validation Error
- `429`: Rate Limit Exceeded
- `500`: Internal Server Error

## Best Practices

1. Implement caching on your end to avoid hitting rate limits
2. Handle rate limit errors gracefully by respecting the reset time
3. Keep text inputs under 100,000 characters
4. Monitor the response headers for rate limit status
5. Implement proper error handling for all possible response codes
