# Vector Databases for RAG

Complete guide to selecting, configuring, and optimizing vector databases for retrieval.

[← Back to Main Guide](../README.md) | [← Previous: Embeddings](./03-embeddings.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Vector Database Comparison](#2-vector-database-comparison)
- [3. Indexing Algorithms](#3-indexing-algorithms)
- [4. Database Selection Guide](#4-database-selection-guide)
- [5. Problem Statements & Solutions](#5-problem-statements--solutions)
- [6. Trade-offs](#6-trade-offs)
- [7. Cost-Effective Solutions](#7-cost-effective-solutions)
- [8. Best Practices](#8-best-practices)
- [9. Quick Reference](#9-quick-reference)

---

## 1. Overview

### What is a Vector Database?

A vector database is specialized storage optimized for storing, indexing, and querying high-dimensional vectors (embeddings).

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Vector Database Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   INGESTION                           QUERYING                      │
│   ─────────                           ────────                      │
│   ┌───────────┐   ┌───────────┐      ┌───────────┐                │
│   │  Vector   │──▶│  Index    │      │   Query   │                │
│   │ + Metadata│   │ (HNSW/IVF)│      │  Vector   │                │
│   └───────────┘   └───────────┘      └─────┬─────┘                │
│                         │                   │                       │
│                         ▼                   ▼                       │
│                   ┌───────────────────────────┐                    │
│                   │     Vector Index           │                    │
│                   │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ │                    │
│                   │  │ ● │ │ ● │ │ ● │ │ ● │ │──▶ Top-K Results   │
│                   │  └───┘ └───┘ └───┘ └───┘ │                    │
│                   └───────────────────────────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Term | Definition |
|------|------------|
| **Vector** | Embedding representation of content |
| **Index** | Data structure for fast similarity search |
| **ANN** | Approximate Nearest Neighbors (fast but not exact) |
| **Recall** | % of true nearest neighbors found |
| **QPS** | Queries Per Second |
| **Metadata** | Additional data stored with vectors |
| **Filtering** | Querying with metadata constraints |

### Why Not Use a Regular Database?

| Operation | Regular DB (B-tree) | Vector DB (HNSW) |
|-----------|---------------------|------------------|
| Find exact match | O(log n) | O(n) |
| Find similar items | O(n) - scan all | O(log n) |
| 1M vectors, top-10 similar | ~1 second | ~1 millisecond |

[↑ Back to Top](#table-of-contents)

---

## 2. Vector Database Comparison

### 2.1 Managed Services

| Database | Best For | Strengths | Limitations | Pricing |
|----------|----------|-----------|-------------|---------|
| **Pinecone** | Quick start, scale | Fully managed, easy | Limited customization | Pod/serverless |
| **Weaviate Cloud** | Hybrid search | GraphQL, modules | Learning curve | Per-vector |
| **Qdrant Cloud** | Performance | Fast, Rust-based | Newer ecosystem | Per-vector |
| **MongoDB Atlas** | Existing Mongo | Unified platform | Vector features basic | Usage-based |
| **Azure AI Search** | Enterprise Azure | Full-text + vector | Microsoft ecosystem | Tier-based |
| **Elastic Cloud** | Log + vector | Mature, full-text | Complex | Node-based |
| **Zilliz Cloud** | Large scale | Milvus managed | Enterprise pricing | Compute units |

### 2.2 Self-Hosted

| Database | Best For | Strengths | Limitations | License |
|----------|----------|-----------|-------------|---------|
| **Milvus** | Large scale | Scalable, feature-rich | Complex setup | Apache 2.0 |
| **Qdrant** | Performance | Fast, rich filtering | Newer | Apache 2.0 |
| **Weaviate** | Hybrid search | Modules, ML-native | Memory usage | BSD-3 |
| **Chroma** | Prototyping | Simple, Python-native | Limited scale | Apache 2.0 |
| **pgvector** | PostgreSQL users | Familiar, ACID | Scale limits | PostgreSQL |
| **Redis Stack** | Caching + vectors | Fast, versatile | Memory-bound | Redis |
| **FAISS** | Research, testing | Meta's library, fast | Not a database | MIT |
| **LanceDB** | Edge/embedded | Serverless, fast | Early stage | Apache 2.0 |

### 2.3 Feature Comparison Matrix

| Feature | Pinecone | Qdrant | Weaviate | Milvus | Chroma | pgvector |
|---------|----------|--------|----------|--------|--------|----------|
| **Managed** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Self-hosted** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Hybrid search** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Metadata filtering** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multi-tenancy** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ACID transactions** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Max vectors** | Billions | Billions | Billions | Billions | Millions | Millions |
| **Horizontal scale** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Learning curve** | Low | Low | Medium | High | Very Low | Low |

[↑ Back to Top](#table-of-contents)

---

## 3. Indexing Algorithms

### 3.1 Algorithm Overview

| Algorithm | Type | Recall | Speed | Memory | Best For |
|-----------|------|--------|-------|--------|----------|
| **Flat** | Exact | 100% | Slow | Low | <10K vectors |
| **IVF** | Cluster | 95-99% | Fast | Medium | Large scale |
| **HNSW** | Graph | 95-99% | Very Fast | High | Most use cases |
| **PQ** | Compression | 90-95% | Fast | Very Low | Memory-constrained |
| **IVF-PQ** | Hybrid | 90-95% | Fast | Low | Large scale, low memory |
| **ScaNN** | Google | 95-99% | Very Fast | Medium | High throughput |

### 3.2 How HNSW Works

```
HNSW (Hierarchical Navigable Small World)
─────────────────────────────────────────

Layer 2 (sparse):     ●─────────────────●
                      │                 │
Layer 1 (medium):     ●───●───●───●─────●
                      │   │   │   │     │
Layer 0 (dense):      ●─●─●─●─●─●─●─●─●─●─●

Search: Start at top layer, navigate down
- Top layers: Long jumps, quick narrowing
- Bottom layers: Fine-grained search
- Result: O(log n) search complexity
```

**HNSW Parameters:**

| Parameter | What it Controls | Default | Tune For |
|-----------|------------------|---------|----------|
| **M** | Connections per node | 16 | More = better recall, more memory |
| **ef_construction** | Build-time search depth | 200 | More = better index, slower build |
| **ef_search** | Query-time search depth | 50 | More = better recall, slower query |

### 3.3 How IVF Works

```
IVF (Inverted File Index)
─────────────────────────

Step 1: Cluster vectors into buckets (centroids)

    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Cluster │  │ Cluster │  │ Cluster │
    │    1    │  │    2    │  │    3    │
    │  ● ● ●  │  │  ● ●    │  │ ● ● ● ● │
    │   ● ●   │  │ ● ● ●   │  │  ● ●    │
    └─────────┘  └─────────┘  └─────────┘

Step 2: Search only nearest clusters (nprobe)

Query ● → Find nearest clusters → Search within them
```

**IVF Parameters:**

| Parameter | What it Controls | Default | Tune For |
|-----------|------------------|---------|----------|
| **nlist** | Number of clusters | sqrt(n) | More = finer partitions |
| **nprobe** | Clusters to search | 10 | More = better recall, slower |

### 3.4 Index Selection Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Index Selection Guide                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  How many vectors?                                                  │
│  ├─► < 10,000      → Flat (exact search)                           │
│  ├─► 10K - 1M      → HNSW (fast, accurate)                         │
│  ├─► 1M - 100M     → IVF or HNSW with tuning                       │
│  └─► > 100M        → IVF-PQ or distributed HNSW                    │
│                                                                     │
│  Memory constrained?                                                │
│  ├─► Yes → IVF-PQ (compressed)                                     │
│  └─► No  → HNSW (fastest)                                          │
│                                                                     │
│  Need exact results?                                                │
│  ├─► Yes → Flat (or HNSW with high ef)                             │
│  └─► No  → HNSW or IVF                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

[↑ Back to Top](#table-of-contents)

---

## 4. Database Selection Guide

### 4.1 Decision Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                Vector Database Selection Guide                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Do you want managed service?                                       │
│  ├─► Yes, minimal ops     → Pinecone                               │
│  ├─► Yes, with features   → Qdrant Cloud or Weaviate Cloud         │
│  └─► No, self-host        → Continue                                │
│                                                                     │
│  Already using PostgreSQL?                                          │
│  ├─► Yes, <1M vectors     → pgvector                               │
│  └─► No                   → Continue                                │
│                                                                     │
│  Need hybrid search (vector + keyword)?                             │
│  ├─► Yes                  → Weaviate or Qdrant                     │
│  └─► No                   → Continue                                │
│                                                                     │
│  Scale requirements?                                                │
│  ├─► < 1M vectors         → Qdrant or Chroma                       │
│  ├─► 1M - 100M           → Qdrant or Milvus                        │
│  └─► > 100M              → Milvus (distributed)                    │
│                                                                     │
│  Just prototyping?                                                  │
│  ├─► Yes                  → Chroma (simplest)                      │
│  └─► No                   → Qdrant (balance of simple + powerful)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation Examples

#### Pinecone

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Get index
index = pc.Index("my-index")

# Upsert vectors
index.upsert(
    vectors=[
        {
            "id": "doc1",
            "values": [0.1, 0.2, ...],  # 1536 dimensions
            "metadata": {"source": "manual.pdf", "page": 1}
        }
    ],
    namespace="my-namespace"
)

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    include_metadata=True,
    filter={"source": {"$eq": "manual.pdf"}}
)
```

#### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize
client = QdrantClient(host="localhost", port=6333)  # or QdrantClient(url="https://...")

# Create collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)

# Upsert vectors
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"source": "manual.pdf", "page": 1}
        )
    ]
)

# Query
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],
    limit=10,
    query_filter={
        "must": [{"key": "source", "match": {"value": "manual.pdf"}}]
    }
)
```

#### Weaviate

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

# Initialize
client = weaviate.connect_to_local()  # or weaviate.connect_to_wcs(...)

# Create collection
collection = client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
    ]
)

# Insert
collection.data.insert(
    properties={"content": "...", "source": "manual.pdf"},
    vector=[0.1, 0.2, ...]
)

# Query
results = collection.query.near_vector(
    near_vector=[0.1, 0.2, ...],
    limit=10,
    filters=weaviate.classes.query.Filter.by_property("source").equal("manual.pdf")
)
```

#### Chroma

```python
import chromadb

# Initialize
client = chromadb.Client()  # or chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    ids=["doc1", "doc2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadatas=[{"source": "manual.pdf"}, {"source": "guide.pdf"}],
    documents=["content 1", "content 2"]  # Optional: store raw text
)

# Query
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=10,
    where={"source": "manual.pdf"}
)
```

#### pgvector

```python
import psycopg2
from pgvector.psycopg2 import register_vector

# Connect
conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)

# Create table
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        source TEXT,
        embedding vector(1536)
    )
""")

# Create index
cur.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
""")

# Insert
cur.execute(
    "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
    ("content", "manual.pdf", [0.1, 0.2, ...])
)

# Query
cur.execute("""
    SELECT content, source, 1 - (embedding <=> %s) as similarity
    FROM documents
    WHERE source = %s
    ORDER BY embedding <=> %s
    LIMIT 10
""", ([0.1, 0.2, ...], "manual.pdf", [0.1, 0.2, ...]))

results = cur.fetchall()
```

[↑ Back to Top](#table-of-contents)

---

## 5. Problem Statements & Solutions

### Problem 1: Slow Query Performance

**Symptoms:**
- Queries take >100ms
- Latency increases with data size
- Timeout errors

**Root Cause:** Index not optimized for query patterns

**Solution:**

```python
# 1. Add HNSW index (most databases)
# Qdrant example:
from qdrant_client.models import HnswConfigDiff

client.update_collection(
    collection_name="my_collection",
    hnsw_config=HnswConfigDiff(
        m=16,                # Connections per node
        ef_construct=200,    # Build-time search depth
    )
)

# 2. Tune search parameters
results = client.search(
    collection_name="my_collection",
    query_vector=query_embedding,
    limit=10,
    search_params={
        "hnsw_ef": 128,  # Higher = better recall, slower
    }
)

# 3. Pre-filter before vector search
results = client.search(
    collection_name="my_collection",
    query_vector=query_embedding,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "technical"}},
            {"key": "date", "range": {"gte": "2024-01-01"}}
        ]
    },
    limit=10
)
```

---

### Problem 2: Poor Recall (Missing Relevant Results)

**Symptoms:**
- Known relevant documents not returned
- Recall metrics are low
- Users complain about missing results

**Root Cause:** Aggressive approximation in index

**Solution:**

```python
# 1. Increase search depth
# For HNSW:
search_params = {"hnsw_ef": 256}  # Default is often 50-100

# For IVF:
search_params = {"nprobe": 50}  # Search more clusters

# 2. Use exact search for small result sets
results = client.search(
    collection_name="my_collection",
    query_vector=query_embedding,
    limit=10,
    search_params={"exact": True}  # Brute force
)

# 3. Increase index quality during build
# Better index = better recall
index_params = {
    "m": 32,                # More connections (default: 16)
    "ef_construction": 400  # More thorough build (default: 200)
}
```

---

### Problem 3: Running Out of Memory

**Symptoms:**
- OOM errors
- Database crashes
- Can't load all vectors

**Root Cause:** HNSW index is memory-intensive

**Solution:**

```python
# 1. Use quantization (reduce vector precision)
# Qdrant example:
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig

client.update_collection(
    collection_name="my_collection",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type="int8",      # 4x memory reduction
            quantile=0.99,
            always_ram=True
        )
    )
)

# 2. Use disk-based index
# Milvus example:
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "DISKANN",  # Disk-based index
        "metric_type": "COSINE",
        "params": {"search_list": 100}
    }
)

# 3. Reduce dimensions (if using matryoshka embeddings)
# Store only first 512 dimensions instead of 1536
truncated_embedding = full_embedding[:512]
```

---

### Problem 4: Inconsistent Results

**Symptoms:**
- Same query returns different results
- Results change after index rebuild
- Non-deterministic behavior

**Root Cause:** Approximate search has randomness; index parameters differ

**Solution:**

```python
# 1. Set random seed for reproducibility (where supported)
import numpy as np
np.random.seed(42)

# 2. Use exact search for critical queries
results = client.search(
    query_vector=embedding,
    search_params={"exact": True}
)

# 3. Increase ef_search for more consistent results
results = client.search(
    query_vector=embedding,
    search_params={"hnsw_ef": 500}  # High value = more consistent
)

# 4. Document and fix index parameters
INDEX_CONFIG = {
    "m": 16,
    "ef_construction": 200,
    "ef_search": 100
}
```

---

### Problem 5: Metadata Filtering is Slow

**Symptoms:**
- Filtered queries much slower than unfiltered
- Post-filter returns few results
- Need to scan many vectors

**Root Cause:** Filter applied after vector search

**Solution:**

```python
# 1. Use pre-filtering (filter first, then vector search)
# Qdrant supports this natively
results = client.search(
    collection_name="my_collection",
    query_vector=embedding,
    query_filter={...},  # Applied before vector search
    limit=10
)

# 2. Create payload indexes for filtered fields
client.create_payload_index(
    collection_name="my_collection",
    field_name="category",
    field_schema="keyword"
)

client.create_payload_index(
    collection_name="my_collection",
    field_name="date",
    field_schema="datetime"
)

# 3. Use partitioning/namespaces for common filters
# Pinecone example - separate namespaces
index.upsert(vectors=tech_docs, namespace="technical")
index.upsert(vectors=legal_docs, namespace="legal")

# Query specific namespace
results = index.query(
    vector=embedding,
    namespace="technical",  # Only searches technical docs
    top_k=10
)
```

---

### Problem 6: Scaling Beyond Single Node

**Symptoms:**
- Single node can't handle data volume
- Query throughput limited
- High availability needed

**Solution:**

```python
# 1. Use managed service (auto-scales)
# Pinecone serverless scales automatically

# 2. Deploy distributed cluster
# Milvus distributed setup
# docker-compose.yml for Milvus cluster
"""
version: '3.5'
services:
  milvus-proxy:
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "proxy"]

  milvus-querynode-1:
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "querynode"]

  milvus-querynode-2:
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "querynode"]

  milvus-datanode:
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "datanode"]
"""

# 3. Implement sharding at application level
class ShardedVectorStore:
    def __init__(self, num_shards: int):
        self.shards = [
            QdrantClient(host=f"qdrant-{i}", port=6333)
            for i in range(num_shards)
        ]

    def get_shard(self, doc_id: str) -> QdrantClient:
        shard_idx = hash(doc_id) % len(self.shards)
        return self.shards[shard_idx]

    def search(self, embedding, limit: int):
        # Query all shards in parallel
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(shard.search, "collection", embedding, limit=limit)
                for shard in self.shards
            ]
            all_results = [f.result() for f in futures]

        # Merge and return top results
        merged = sorted(
            [r for results in all_results for r in results],
            key=lambda x: x.score,
            reverse=True
        )
        return merged[:limit]
```

[↑ Back to Top](#table-of-contents)

---

## 6. Trade-offs

### Managed vs Self-Hosted

| Factor | Managed | Self-Hosted |
|--------|---------|-------------|
| **Setup time** | ✅ Minutes | ❌ Hours/Days |
| **Maintenance** | ✅ None | ❌ Ongoing |
| **Cost (low volume)** | ❌ Minimum fees | ✅ Just compute |
| **Cost (high volume)** | ❌ Can be high | ✅ More control |
| **Customization** | ❌ Limited | ✅ Full control |
| **Data residency** | ⚠️ Provider regions | ✅ Your choice |
| **Scaling** | ✅ Automatic | ❌ Manual |

### Index Type Trade-offs

| Index | Build Time | Query Speed | Memory | Recall |
|-------|------------|-------------|--------|--------|
| Flat | O(n) | O(n) | 1x | 100% |
| HNSW | O(n log n) | O(log n) | 2-4x | 95-99% |
| IVF | O(n) | O(√n) | 1.1x | 90-98% |
| PQ | O(n) | O(n/8) | 0.1x | 85-95% |

### Database Trade-offs

| Database | Ease of Use | Performance | Scale | Cost |
|----------|-------------|-------------|-------|------|
| Pinecone | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ |
| Qdrant | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$ |
| Weaviate | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$ |
| Milvus | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $ |
| Chroma | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Free |
| pgvector | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | $ |

[↑ Back to Top](#table-of-contents)

---

## 7. Cost-Effective Solutions

### Cost Comparison (1M vectors, 1536 dimensions)

| Solution | Monthly Cost | Notes |
|----------|--------------|-------|
| **Pinecone Serverless** | ~$70 | Usage-based |
| **Pinecone Pods (p1)** | ~$70 | Fixed capacity |
| **Qdrant Cloud** | ~$30 | Per-vector pricing |
| **Weaviate Cloud** | ~$25 | Sandbox free |
| **Self-hosted Qdrant** | ~$20 | 2 vCPU, 4GB RAM |
| **pgvector on RDS** | ~$30 | db.t3.medium |
| **Chroma (local)** | $0 | Development only |

### Free Options

```python
# 1. Chroma (local, free, great for development)
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")

# 2. pgvector on local Postgres
# Just install the extension
# CREATE EXTENSION vector;

# 3. FAISS (not a database, but free for search)
import faiss
index = faiss.IndexFlatIP(1536)  # Inner product
index.add(vectors)
distances, indices = index.search(query_vector, k=10)

# 4. LanceDB (embedded, serverless, free)
import lancedb
db = lancedb.connect("./lance_db")
table = db.create_table("documents", data)
```

### Cost Optimization Strategies

```python
# 1. Use quantization to reduce storage
# Reduces memory by 4x
client.update_collection(
    collection_name="my_collection",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type="int8")
    )
)

# 2. Use matryoshka embeddings with lower dimensions
# 512 dims instead of 1536 = 3x less storage
truncated = embedding[:512]

# 3. Archive old data to cheaper storage
# Keep hot data in vector DB, cold data in object storage

# 4. Use namespaces/partitions to query less data
# Only query relevant subset

# 5. Implement caching for frequent queries
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_search(query_hash):
    return vector_store.search(query_embedding)
```

[↑ Back to Top](#table-of-contents)

---

## 8. Best Practices

### DO's

1. **Choose the Right Index for Your Scale**
   ```python
   # < 10K vectors: Flat is fine
   # 10K - 1M: HNSW
   # > 1M: IVF or distributed HNSW
   ```

2. **Index Frequently Filtered Fields**
   ```python
   # Create indexes for metadata filters
   client.create_payload_index(
       collection_name="docs",
       field_name="category",
       field_schema="keyword"
   )
   ```

3. **Use Namespaces/Partitions for Multi-tenancy**
   ```python
   # Pinecone
   index.upsert(vectors, namespace=f"tenant_{tenant_id}")

   # Qdrant
   client.create_collection(f"tenant_{tenant_id}", ...)
   ```

4. **Monitor Key Metrics**
   ```python
   metrics_to_track = [
       "query_latency_p99",
       "indexing_throughput",
       "memory_usage",
       "recall_at_10",
       "vectors_count"
   ]
   ```

5. **Implement Backup Strategy**
   ```python
   # Qdrant snapshot
   client.create_snapshot(collection_name="my_collection")

   # Regular exports
   def backup_collection(collection_name):
       points = client.scroll(collection_name, limit=10000)
       save_to_storage(points)
   ```

### DON'Ts

1. **Don't Use Flat Index at Scale**
   - O(n) search is too slow for >10K vectors

2. **Don't Ignore Metadata Indexes**
   - Filtering without indexes scans all vectors

3. **Don't Store Raw Text in Vector DB**
   - Store only IDs, keep text in document store

4. **Don't Skip Capacity Planning**
   - Estimate memory: vectors × dimensions × 4 bytes × overhead

5. **Don't Forget About Updates**
   - Plan for document updates and deletes

[↑ Back to Top](#table-of-contents)

---

## 9. Quick Reference

### Database Selection Cheat Sheet

```
Requirement → Database
─────────────────────────────────────
Quickest start        → Pinecone or Chroma
Lowest cost           → Chroma (free) or pgvector
Highest performance   → Qdrant
Largest scale         → Milvus
Hybrid search         → Weaviate or Qdrant
PostgreSQL user       → pgvector
Azure enterprise      → Azure AI Search
```

### Index Selection Cheat Sheet

```
Vector Count → Index Type
─────────────────────────────────────
< 10K         → Flat (exact)
10K - 100K    → HNSW (default)
100K - 10M    → HNSW (tuned) or IVF
> 10M         → IVF-PQ or distributed
Memory tight  → IVF-PQ or quantized HNSW
```

### HNSW Tuning Cheat Sheet

```
Scenario → Parameters
─────────────────────────────────────
Default         → m=16, ef_c=200, ef_s=50
Higher recall   → m=32, ef_c=400, ef_s=200
Lower memory    → m=8, ef_c=100, ef_s=50
Faster queries  → m=16, ef_c=200, ef_s=20
```

### Memory Estimation

```python
def estimate_memory(num_vectors: int, dimensions: int, index_type: str) -> str:
    """Estimate memory requirements."""

    base_memory = num_vectors * dimensions * 4  # 4 bytes per float

    multipliers = {
        "flat": 1.0,
        "hnsw": 1.5,  # Graph overhead
        "ivf": 1.1,
        "pq": 0.1,    # Compressed
    }

    total_bytes = base_memory * multipliers.get(index_type, 1.5)

    if total_bytes < 1e9:
        return f"{total_bytes / 1e6:.1f} MB"
    else:
        return f"{total_bytes / 1e9:.1f} GB"

# Examples:
# 1M vectors, 1536 dims, HNSW = ~9.2 GB
# 1M vectors, 1536 dims, PQ = ~0.6 GB
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Embeddings](./03-embeddings.md) | [Main Guide](../README.md) | [Retrieval →](./05-retrieval.md) |

---

*Last updated: 2024*
