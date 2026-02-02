# Production Deployment for RAG

Complete guide to deploying, scaling, and operating RAG systems in production.

[← Back to Main Guide](../README.md) | [← Previous: Security & Compliance](./08-security-compliance.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Architecture Patterns](#2-architecture-patterns)
- [3. Infrastructure Setup](#3-infrastructure-setup)
- [4. Scaling Strategies](#4-scaling-strategies)
- [5. Caching Strategies](#5-caching-strategies)
- [6. Error Handling & Resilience](#6-error-handling--resilience)
- [7. Problem Statements & Solutions](#7-problem-statements--solutions)
- [8. Trade-offs](#8-trade-offs)
- [9. Cost-Effective Solutions](#9-cost-effective-solutions)
- [10. Best Practices](#10-best-practices)
- [11. Quick Reference](#11-quick-reference)

---

## 1. Overview

### Production Requirements

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Production RAG Requirements                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RELIABILITY           PERFORMANCE           SCALABILITY           │
│  ───────────           ───────────           ───────────           │
│  • 99.9% uptime        • <500ms p95          • Handle 10x load    │
│  • Graceful failures   • Consistent latency  • Horizontal scale   │
│  • Auto-recovery       • High throughput     • Auto-scaling       │
│                                                                     │
│  OBSERVABILITY         SECURITY              COST                  │
│  ────────────          ────────              ────                  │
│  • Full tracing        • Auth/authz          • Optimized usage    │
│  • Metrics & alerts    • Data protection     • Predictable costs  │
│  • Log aggregation     • Audit trails        • Resource efficiency│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Production vs Development

| Aspect | Development | Production |
|--------|-------------|------------|
| **Vector DB** | Chroma local | Managed Qdrant/Pinecone |
| **LLM** | Single API key | Load balanced, fallbacks |
| **Caching** | None | Multi-layer caching |
| **Monitoring** | Console logs | Full observability stack |
| **Scaling** | Single instance | Auto-scaling cluster |
| **Security** | Basic auth | Full RBAC, encryption |

[↑ Back to Top](#table-of-contents)

---

## 2. Architecture Patterns

### 2.1 Simple Production Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Simple Production RAG                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│     Users                                                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                   │
│  │   API       │  ──── Auth ────▶  Identity Provider               │
│  │   Gateway   │                                                   │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │    RAG      │────▶│   Vector    │     │    LLM      │          │
│  │   Service   │────▶│     DB      │────▶│    API      │          │
│  └──────┬──────┘     └─────────────┘     └─────────────┘          │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                   │
│  │   Cache     │                                                   │
│  │   (Redis)   │                                                   │
│  └─────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Scalable Production Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Scalable Production RAG                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                      ┌─────────────────┐                           │
│                      │  Load Balancer  │                           │
│                      └────────┬────────┘                           │
│                               │                                     │
│            ┌──────────────────┼──────────────────┐                 │
│            │                  │                  │                 │
│     ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐        │
│     │  RAG API    │    │  RAG API    │    │  RAG API    │        │
│     │  Instance 1 │    │  Instance 2 │    │  Instance N │        │
│     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│            │                  │                  │                 │
│            └──────────────────┼──────────────────┘                 │
│                               │                                     │
│     ┌─────────────────────────┼─────────────────────────┐         │
│     │                         │                         │         │
│     ▼                         ▼                         ▼         │
│  ┌──────────┐          ┌──────────┐          ┌──────────┐        │
│  │  Redis   │          │ Vector   │          │  LLM     │        │
│  │  Cache   │          │ DB       │          │  Router  │        │
│  │  Cluster │          │ Cluster  │          │          │        │
│  └──────────┘          └──────────┘          └────┬─────┘        │
│                                                   │               │
│                              ┌────────────────────┼────────┐     │
│                              │                    │        │     │
│                              ▼                    ▼        ▼     │
│                         ┌────────┐          ┌────────┐ ┌────────┐│
│                         │ OpenAI │          │ Claude │ │ Local  ││
│                         └────────┘          └────────┘ │ LLM    ││
│                                                        └────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Microservices Architecture

```python
# Service definitions for microservices RAG

# 1. Query Service
class QueryService:
    """Handle incoming queries."""

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        # Validate
        validated = await self.validator.validate(request)

        # Route to retrieval service
        retrieval_result = await self.retrieval_client.retrieve(
            query=validated.query,
            user_context=validated.user
        )

        # Route to generation service
        response = await self.generation_client.generate(
            query=validated.query,
            context=retrieval_result.documents
        )

        return QueryResponse(
            answer=response.text,
            sources=retrieval_result.sources
        )

# 2. Retrieval Service
class RetrievalService:
    """Handle document retrieval."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        # Embed query
        embedding = await self.embedding_service.embed(request.query)

        # Search vector DB
        results = await self.vector_db.search(
            embedding=embedding,
            filter=request.filter,
            top_k=request.top_k
        )

        # Re-rank
        reranked = await self.reranker.rerank(request.query, results)

        return RetrievalResponse(documents=reranked)

# 3. Generation Service
class GenerationService:
    """Handle response generation."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Build prompt
        prompt = self.prompt_builder.build(
            query=request.query,
            context=request.context
        )

        # Call LLM with fallback
        response = await self.llm_router.generate(prompt)

        return GenerationResponse(text=response)

# 4. Embedding Service
class EmbeddingService:
    """Handle text embedding."""

    async def embed(self, text: str) -> list[float]:
        # Check cache
        cached = await self.cache.get(text)
        if cached:
            return cached

        # Generate embedding
        embedding = await self.model.encode(text)

        # Cache
        await self.cache.set(text, embedding)

        return embedding
```

[↑ Back to Top](#table-of-contents)

---

## 3. Infrastructure Setup

### 3.1 Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - VECTOR_DB_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1'

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api

volumes:
  qdrant_data:
  redis_data:
```

### 3.2 Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  labels:
    app: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: VECTOR_DB_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: vector-db-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3.3 FastAPI Application Structure

```python
# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from app.config import Settings
from app.services import RAGService
from app.middleware import RateLimitMiddleware, LoggingMiddleware

logger = structlog.get_logger()
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting RAG application")
    app.state.rag_service = RAGService(settings)
    await app.state.rag_service.initialize()

    yield

    # Shutdown
    logger.info("Shutting down RAG application")
    await app.state.rag_service.cleanup()

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)

# Health endpoints
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    # Check dependencies
    rag_service = app.state.rag_service
    if not await rag_service.is_ready():
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

# Query endpoint
@app.post("/query")
async def query(request: QueryRequest, user: User = Depends(get_current_user)):
    rag_service = app.state.rag_service

    try:
        result = await rag_service.query(
            query=request.query,
            user=user,
            options=request.options
        )
        return result
    except Exception as e:
        logger.error("Query failed", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail="Query processing failed")
```

[↑ Back to Top](#table-of-contents)

---

## 4. Scaling Strategies

### 4.1 Horizontal Scaling

```python
class ScalableRAGService:
    """RAG service designed for horizontal scaling."""

    def __init__(self, config: Config):
        self.config = config

        # Stateless - all state in external services
        self.vector_db = VectorDBClient(config.vector_db_url)
        self.cache = RedisClient(config.redis_url)
        self.llm_client = LLMClient(config.llm_api_key)

        # Instance ID for tracking
        self.instance_id = os.environ.get("HOSTNAME", str(uuid.uuid4())[:8])

    async def query(self, query: str, user: dict) -> dict:
        """Stateless query processing."""

        # All state is external - can run on any instance
        start_time = time.time()

        # 1. Check cache (shared across instances)
        cache_key = self.build_cache_key(query, user)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        # 2. Retrieve (stateless call to vector DB)
        results = await self.vector_db.search(
            query_vector=await self.embed(query),
            filter=self.build_filter(user)
        )

        # 3. Generate (stateless call to LLM)
        response = await self.llm_client.generate(
            prompt=self.build_prompt(query, results)
        )

        # 4. Cache result
        result = {"response": response, "sources": results}
        await self.cache.set(cache_key, result, ttl=3600)

        # 5. Log metrics
        self.log_metrics(query, time.time() - start_time)

        return result
```

### 4.2 Load Balancing LLM Calls

```python
class LLMLoadBalancer:
    """Load balance across multiple LLM providers."""

    def __init__(self):
        self.providers = [
            {"name": "openai", "client": OpenAIClient(), "weight": 0.5, "healthy": True},
            {"name": "anthropic", "client": AnthropicClient(), "weight": 0.3, "healthy": True},
            {"name": "local", "client": LocalLLMClient(), "weight": 0.2, "healthy": True},
        ]

        self.circuit_breakers = {
            p["name"]: CircuitBreaker(failure_threshold=3, recovery_timeout=60)
            for p in self.providers
        }

    async def generate(self, prompt: str) -> str:
        """Generate with load balancing and fallback."""

        # Get healthy providers
        healthy = [p for p in self.providers if p["healthy"]]

        if not healthy:
            raise Exception("No healthy LLM providers")

        # Weighted random selection
        provider = self.weighted_select(healthy)

        try:
            # Check circuit breaker
            if self.circuit_breakers[provider["name"]].is_open():
                raise CircuitOpenError()

            # Make request
            response = await provider["client"].generate(prompt)

            # Success - record
            self.circuit_breakers[provider["name"]].record_success()

            return response

        except Exception as e:
            # Failure - record and try fallback
            self.circuit_breakers[provider["name"]].record_failure()

            # Try next provider
            return await self.generate_with_fallback(prompt, exclude=provider["name"])

    def weighted_select(self, providers: list) -> dict:
        """Select provider based on weights."""
        import random

        total_weight = sum(p["weight"] for p in providers)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for provider in providers:
            cumulative += provider["weight"]
            if r <= cumulative:
                return provider

        return providers[-1]

    async def generate_with_fallback(self, prompt: str, exclude: str) -> str:
        """Try other providers as fallback."""

        for provider in self.providers:
            if provider["name"] == exclude:
                continue
            if not provider["healthy"]:
                continue

            try:
                return await provider["client"].generate(prompt)
            except:
                continue

        raise Exception("All LLM providers failed")
```

### 4.3 Auto-Scaling Configuration

```python
class AutoScalingManager:
    """Manage auto-scaling based on metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics_client = MetricsClient()

    def get_scaling_decision(self) -> dict:
        """Determine if scaling is needed."""

        metrics = self.metrics_client.get_current_metrics()

        decision = {
            "action": "none",
            "reason": "",
            "target_replicas": None
        }

        # Scale up conditions
        if metrics["cpu_utilization"] > 80:
            decision["action"] = "scale_up"
            decision["reason"] = f"CPU at {metrics['cpu_utilization']}%"
        elif metrics["memory_utilization"] > 85:
            decision["action"] = "scale_up"
            decision["reason"] = f"Memory at {metrics['memory_utilization']}%"
        elif metrics["request_latency_p95"] > 2000:
            decision["action"] = "scale_up"
            decision["reason"] = f"P95 latency at {metrics['request_latency_p95']}ms"
        elif metrics["queue_depth"] > 100:
            decision["action"] = "scale_up"
            decision["reason"] = f"Queue depth at {metrics['queue_depth']}"

        # Scale down conditions
        elif metrics["cpu_utilization"] < 30 and metrics["memory_utilization"] < 40:
            decision["action"] = "scale_down"
            decision["reason"] = "Low resource utilization"

        return decision

    def calculate_target_replicas(self, current: int, decision: dict) -> int:
        """Calculate target number of replicas."""

        if decision["action"] == "scale_up":
            return min(current + 2, self.config.max_replicas)
        elif decision["action"] == "scale_down":
            return max(current - 1, self.config.min_replicas)
        return current
```

[↑ Back to Top](#table-of-contents)

---

## 5. Caching Strategies

### 5.1 Multi-Layer Caching

```python
class MultiLayerCache:
    """Multi-layer caching for RAG."""

    def __init__(self, redis_client, local_cache_size: int = 1000):
        self.redis = redis_client
        self.local = LRUCache(maxsize=local_cache_size)

        self.layers = [
            {"name": "local", "get": self.local_get, "set": self.local_set, "ttl": 300},
            {"name": "redis", "get": self.redis_get, "set": self.redis_set, "ttl": 3600},
        ]

    async def get(self, key: str) -> any:
        """Get from cache, checking each layer."""

        for layer in self.layers:
            value = await layer["get"](key)
            if value is not None:
                # Populate higher layers
                await self.populate_higher_layers(key, value, layer)
                return value

        return None

    async def set(self, key: str, value: any, ttl: int = None):
        """Set in all cache layers."""

        for layer in self.layers:
            layer_ttl = ttl or layer["ttl"]
            await layer["set"](key, value, layer_ttl)

    async def populate_higher_layers(self, key: str, value: any, source_layer: dict):
        """Populate cache layers above the source."""

        for layer in self.layers:
            if layer["name"] == source_layer["name"]:
                break
            await layer["set"](key, value, layer["ttl"])

    # Layer implementations
    async def local_get(self, key: str) -> any:
        return self.local.get(key)

    async def local_set(self, key: str, value: any, ttl: int):
        self.local[key] = value

    async def redis_get(self, key: str) -> any:
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def redis_set(self, key: str, value: any, ttl: int):
        await self.redis.setex(key, ttl, json.dumps(value))
```

### 5.2 Semantic Caching

```python
class SemanticCache:
    """Cache similar queries, not just exact matches."""

    def __init__(self, embedding_model, vector_store, similarity_threshold: float = 0.95):
        self.embedder = embedding_model
        self.cache_store = vector_store
        self.threshold = similarity_threshold

    async def get(self, query: str) -> dict | None:
        """Find cached response for similar query."""

        # Embed query
        query_embedding = await self.embedder.encode(query)

        # Search for similar cached queries
        results = await self.cache_store.search(
            query_vector=query_embedding,
            limit=1
        )

        if results and results[0]["score"] >= self.threshold:
            cached = results[0]
            return {
                "response": cached["metadata"]["response"],
                "original_query": cached["metadata"]["query"],
                "similarity": cached["score"],
                "cached": True
            }

        return None

    async def set(self, query: str, response: str, ttl: int = 3600):
        """Cache query-response pair."""

        query_embedding = await self.embedder.encode(query)

        await self.cache_store.upsert(
            id=f"cache_{hash(query)}",
            vector=query_embedding,
            metadata={
                "query": query,
                "response": response,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()
            }
        )

    async def invalidate_expired(self):
        """Remove expired cache entries."""
        now = datetime.utcnow().isoformat()
        await self.cache_store.delete(
            filter={"expires_at": {"$lt": now}}
        )
```

### 5.3 Embedding Cache

```python
class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, redis_client, embedding_model):
        self.redis = redis_client
        self.model = embedding_model
        self.prefix = "emb:"

    async def get_or_compute(self, text: str) -> list[float]:
        """Get embedding from cache or compute."""

        cache_key = self.build_key(text)

        # Try cache
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Compute
        embedding = self.model.encode(text).tolist()

        # Cache (embeddings don't expire - text->embedding is deterministic)
        await self.redis.set(cache_key, json.dumps(embedding))

        return embedding

    async def get_or_compute_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch get/compute embeddings."""

        results = [None] * len(texts)
        to_compute = []
        to_compute_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self.build_key(text)
            cached = await self.redis.get(cache_key)

            if cached:
                results[i] = json.loads(cached)
            else:
                to_compute.append(text)
                to_compute_indices.append(i)

        # Batch compute missing
        if to_compute:
            computed = self.model.encode(to_compute).tolist()

            # Store results and cache
            for idx, text, embedding in zip(to_compute_indices, to_compute, computed):
                results[idx] = embedding
                cache_key = self.build_key(text)
                await self.redis.set(cache_key, json.dumps(embedding))

        return results

    def build_key(self, text: str) -> str:
        """Build cache key from text."""
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{self.prefix}{text_hash}"
```

[↑ Back to Top](#table-of-contents)

---

## 6. Error Handling & Resilience

### 6.1 Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return False
            return True
        return False

    def record_success(self):
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0

    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.is_open():
            raise CircuitOpenError("Circuit is open")

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
```

### 6.2 Retry with Exponential Backoff

```python
import asyncio
import random

class RetryHandler:
    """Handle retries with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    async def execute(self, func, *args, **kwargs):
        """Execute function with retries."""

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except RetryableError as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    await asyncio.sleep(delay)

            except NonRetryableError:
                raise

        raise last_exception

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""

        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay

# Retryable errors
class RetryableError(Exception):
    """Error that can be retried."""
    pass

class NonRetryableError(Exception):
    """Error that should not be retried."""
    pass
```

### 6.3 Graceful Degradation

```python
class GracefulDegradation:
    """Handle degraded operation modes."""

    def __init__(self, rag_service):
        self.rag = rag_service
        self.degradation_mode = "normal"

    async def query(self, query: str, user: dict) -> dict:
        """Query with graceful degradation."""

        try:
            # Try full functionality
            if self.degradation_mode == "normal":
                return await self.full_query(query, user)

            # Degraded modes
            elif self.degradation_mode == "no_rerank":
                return await self.query_without_rerank(query, user)

            elif self.degradation_mode == "cached_only":
                return await self.cached_only_query(query, user)

            elif self.degradation_mode == "maintenance":
                return self.maintenance_response()

        except VectorDBUnavailable:
            self.degradation_mode = "cached_only"
            return await self.cached_only_query(query, user)

        except LLMUnavailable:
            return self.llm_unavailable_response(query)

    async def full_query(self, query: str, user: dict) -> dict:
        """Full RAG query with all features."""
        return await self.rag.query(query, user)

    async def query_without_rerank(self, query: str, user: dict) -> dict:
        """Query without re-ranking (faster, lower quality)."""
        return await self.rag.query(query, user, skip_rerank=True)

    async def cached_only_query(self, query: str, user: dict) -> dict:
        """Only return cached responses."""
        cached = await self.rag.cache.get(query)
        if cached:
            cached["degraded"] = True
            return cached
        return {
            "error": "Service temporarily degraded. Please try again later.",
            "degraded": True
        }

    def maintenance_response(self) -> dict:
        return {
            "error": "System is under maintenance. Please try again later.",
            "maintenance": True
        }

    def llm_unavailable_response(self, query: str) -> dict:
        return {
            "error": "AI service temporarily unavailable.",
            "suggestion": "Here are relevant documents you can review:",
            "documents": self.get_fallback_documents(query),
            "degraded": True
        }
```

### 6.4 Health Checks

```python
class HealthChecker:
    """Comprehensive health checks."""

    def __init__(self, services: dict):
        self.services = services

    async def check_all(self) -> dict:
        """Check health of all services."""

        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }

        for name, checker in self.services.items():
            try:
                service_health = await checker()
                results["services"][name] = service_health
            except Exception as e:
                results["services"][name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                results["status"] = "degraded"

        # Determine overall status
        unhealthy_count = sum(
            1 for s in results["services"].values()
            if s.get("status") == "unhealthy"
        )

        if unhealthy_count == len(self.services):
            results["status"] = "unhealthy"
        elif unhealthy_count > 0:
            results["status"] = "degraded"

        return results

    async def check_vector_db(self) -> dict:
        """Check vector database health."""
        start = time.time()
        try:
            # Ping or simple query
            await self.vector_db.ping()
            return {
                "status": "healthy",
                "latency_ms": (time.time() - start) * 1000
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_llm(self) -> dict:
        """Check LLM API health."""
        start = time.time()
        try:
            # Small test generation
            await self.llm.generate("test", max_tokens=1)
            return {
                "status": "healthy",
                "latency_ms": (time.time() - start) * 1000
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_cache(self) -> dict:
        """Check cache health."""
        try:
            await self.cache.ping()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
```

[↑ Back to Top](#table-of-contents)

---

## 7. Problem Statements & Solutions

### Problem 1: Cold Start Latency

**Solution:**

```python
class WarmupManager:
    """Handle cold start warming."""

    def __init__(self, rag_service):
        self.rag = rag_service

    async def warmup(self):
        """Warm up the service on startup."""

        tasks = [
            self.warmup_embeddings(),
            self.warmup_vector_db(),
            self.warmup_llm(),
            self.warmup_cache()
        ]

        await asyncio.gather(*tasks)

    async def warmup_embeddings(self):
        """Load embedding model into memory."""
        # Force model loading
        _ = self.rag.embedding_model.encode("warmup query")

    async def warmup_vector_db(self):
        """Warm up vector DB connection pool."""
        # Run a simple query
        await self.rag.vector_db.search(
            query_vector=[0.0] * 1536,
            limit=1
        )

    async def warmup_llm(self):
        """Establish LLM API connection."""
        await self.rag.llm_client.generate("Hello", max_tokens=1)

    async def warmup_cache(self):
        """Warm up cache with common queries."""
        common_queries = await self.get_common_queries()
        for query in common_queries[:10]:
            try:
                await self.rag.query(query, user={"id": "warmup"})
            except:
                pass
```

---

### Problem 2: Inconsistent Response Times

**Solution:**

```python
class LatencyOptimizer:
    """Optimize for consistent latency."""

    def __init__(self):
        self.timeout_ms = 5000  # 5 second timeout
        self.early_termination_ms = 3000  # Start degrading at 3s

    async def query_with_timeout(self, rag_service, query: str) -> dict:
        """Query with timeout and early termination."""

        start_time = time.time()

        # Start retrieval
        retrieval_task = asyncio.create_task(
            rag_service.retrieve(query)
        )

        # Wait for retrieval with timeout
        try:
            results = await asyncio.wait_for(
                retrieval_task,
                timeout=self.early_termination_ms / 1000
            )
        except asyncio.TimeoutError:
            # Use cached/fallback results
            results = await self.get_fallback_results(query)

        elapsed = (time.time() - start_time) * 1000
        remaining = self.timeout_ms - elapsed

        # Start generation with remaining time budget
        if remaining > 1000:  # At least 1 second for generation
            try:
                response = await asyncio.wait_for(
                    rag_service.generate(query, results),
                    timeout=remaining / 1000
                )
            except asyncio.TimeoutError:
                response = self.generate_fallback_response(results)
        else:
            response = self.generate_fallback_response(results)

        return {"response": response, "sources": results}
```

[↑ Back to Top](#table-of-contents)

---

## 8. Trade-offs

### Scaling Trade-offs

| Approach | Complexity | Cost | Performance |
|----------|------------|------|-------------|
| Vertical scaling | Low | Medium | Limited |
| Horizontal scaling | Medium | Variable | High |
| Serverless | Low | Variable | Variable |

### Caching Trade-offs

| Strategy | Hit Rate | Freshness | Memory |
|----------|----------|-----------|--------|
| Exact match | Low | High | Low |
| Semantic cache | High | Medium | High |
| TTL-based | Medium | Configurable | Medium |

[↑ Back to Top](#table-of-contents)

---

## 9. Cost-Effective Solutions

### Infrastructure Cost Optimization

```python
class CostOptimizer:
    """Optimize infrastructure costs."""

    def __init__(self):
        self.recommendations = []

    def analyze_usage(self, metrics: dict) -> list:
        """Analyze usage and provide recommendations."""

        # Check for over-provisioning
        if metrics["avg_cpu"] < 30:
            self.recommendations.append({
                "type": "downsize",
                "component": "compute",
                "current": metrics["instance_type"],
                "recommended": self.get_smaller_instance(metrics["instance_type"]),
                "savings": "~30%"
            })

        # Check cache hit rate
        if metrics["cache_hit_rate"] < 0.3:
            self.recommendations.append({
                "type": "improve_caching",
                "current_hit_rate": metrics["cache_hit_rate"],
                "suggestion": "Implement semantic caching",
                "potential_savings": "20-40% LLM costs"
            })

        return self.recommendations
```

### LLM Cost Optimization

```
Strategy                    Savings    Trade-off
─────────────────────────────────────────────────
Smaller model for simple Q  50-80%    Lower quality
Semantic caching            30-50%    Stale responses
Prompt compression          20-30%    Context loss
Batch processing            10-20%    Latency
```

[↑ Back to Top](#table-of-contents)

---

## 10. Best Practices

### Deployment Checklist

```
□ Health checks configured
□ Graceful shutdown implemented
□ Resource limits set
□ Auto-scaling configured
□ Monitoring and alerts set up
□ Backup and recovery tested
□ Security hardened
□ Load testing completed
□ Runbook documented
```

### DO's

1. **Use health checks for all dependencies**
2. **Implement circuit breakers for external calls**
3. **Cache at multiple levels**
4. **Monitor and alert on key metrics**
5. **Plan for graceful degradation**

### DON'Ts

1. **Don't deploy without load testing**
2. **Don't skip health checks**
3. **Don't ignore error rates**
4. **Don't hardcode configuration**
5. **Don't skip security review**

[↑ Back to Top](#table-of-contents)

---

## 11. Quick Reference

### Key Metrics to Monitor

```
Metric              Warning    Critical
─────────────────────────────────────────
P95 Latency         > 2s       > 5s
Error Rate          > 1%       > 5%
CPU Utilization     > 70%      > 90%
Memory Usage        > 75%      > 90%
Cache Hit Rate      < 30%      < 10%
```

### Infrastructure Sizing Guide

```
Scale              Compute         Vector DB        Cache
─────────────────────────────────────────────────────────
Small (<1K/day)    2 vCPU, 4GB    Managed small    1GB Redis
Medium (<10K/day)  4 vCPU, 8GB    Managed medium   4GB Redis
Large (<100K/day)  8+ vCPU, 16GB  Managed large    16GB Redis
Enterprise         Auto-scale      Distributed      Redis Cluster
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Security & Compliance](./08-security-compliance.md) | [Main Guide](../README.md) | [Advanced Patterns →](./10-advanced-patterns.md) |

---

*Last updated: 2024*
