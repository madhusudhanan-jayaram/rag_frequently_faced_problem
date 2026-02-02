# Embedding Models for RAG

Complete guide to selecting, using, and optimizing embedding models for retrieval.

[← Back to Main Guide](../README.md) | [← Previous: Chunking](./02-chunking.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Embedding Model Comparison](#2-embedding-model-comparison)
- [3. Model Selection Guide](#3-model-selection-guide)
- [4. Fine-tuning Embeddings](#4-fine-tuning-embeddings)
- [5. Problem Statements & Solutions](#5-problem-statements--solutions)
- [6. Trade-offs](#6-trade-offs)
- [7. Cost-Effective Solutions](#7-cost-effective-solutions)
- [8. Best Practices](#8-best-practices)
- [9. Quick Reference](#9-quick-reference)

---

## 1. Overview

### What Are Embeddings?

Embeddings are numerical representations (vectors) of text that capture semantic meaning. Similar texts have similar vectors, enabling semantic search.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    How Embeddings Work                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Text Input                    Embedding Model                      │
│  ───────────                   ───────────────                      │
│  "How do I reset              ┌─────────────┐     [0.023, -0.156,  │
│   my password?"          ───▶ │  Encoder    │ ──▶  0.847, 0.234,  │
│                               └─────────────┘      ..., 0.091]    │
│                                                    (1536 dimensions)│
│                                                                     │
│  Similar Meaning → Similar Vectors                                  │
│  ────────────────────────────────                                  │
│  "password reset"        ──▶  [0.021, -0.148, 0.851, ...]  ✓ Close │
│  "change my credentials" ──▶  [0.019, -0.161, 0.839, ...]  ✓ Close │
│  "buy a new laptop"      ──▶  [0.412, 0.089, -0.234, ...]  ✗ Far   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Term | Definition |
|------|------------|
| **Embedding** | Vector representation of text |
| **Dimensions** | Number of values in the vector (e.g., 1536) |
| **Similarity** | How close two vectors are (cosine, dot product) |
| **Bi-encoder** | Encodes query and document separately |
| **Cross-encoder** | Encodes query-document pairs together (for re-ranking) |

### Embedding Pipeline

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Chunk   │──▶│ Tokenize │──▶│  Encode  │──▶│  Store   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                    │              │              │
                    ▼              ▼              ▼
               Split into     Transform      Save to
               tokens         to vector      vector DB
```

[↑ Back to Top](#table-of-contents)

---

## 2. Embedding Model Comparison

### 2.1 Proprietary Models

| Model | Provider | Dimensions | Max Tokens | Cost/1M Tokens | MTEB Rank |
|-------|----------|------------|------------|----------------|-----------|
| **text-embedding-3-large** | OpenAI | 3072 | 8191 | $0.13 | Top 10 |
| **text-embedding-3-small** | OpenAI | 1536 | 8191 | $0.02 | Top 30 |
| **text-embedding-ada-002** | OpenAI | 1536 | 8191 | $0.10 | Legacy |
| **embed-english-v3.0** | Cohere | 1024 | 512 | $0.10 | Top 15 |
| **embed-multilingual-v3.0** | Cohere | 1024 | 512 | $0.10 | Top 5 (multilingual) |
| **voyage-large-2** | Voyage AI | 1536 | 16000 | $0.12 | Top 5 |
| **voyage-code-2** | Voyage AI | 1536 | 16000 | $0.12 | #1 (code) |
| **text-embedding-004** | Google | 768 | 2048 | Usage-based | Top 20 |
| **titan-embed-text-v2** | AWS | 1024 | 8000 | $0.02 | Top 25 |

**OpenAI Implementation:**

```python
from openai import OpenAI

client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed texts using OpenAI."""

    response = client.embeddings.create(
        input=texts,
        model=model
    )

    return [item.embedding for item in response.data]

# With dimension reduction (text-embedding-3 only)
def embed_with_dimensions(texts: list[str], dimensions: int = 512) -> list[list[float]]:
    """Embed with reduced dimensions for cost/speed."""

    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
        dimensions=dimensions  # 256, 512, 1024, 1536
    )

    return [item.embedding for item in response.data]
```

**Cohere Implementation:**

```python
import cohere

co = cohere.Client("your-api-key")

def embed_cohere(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """
    Embed texts using Cohere.

    input_type options:
    - "search_document": For documents being indexed
    - "search_query": For search queries
    - "classification": For classification tasks
    - "clustering": For clustering tasks
    """

    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type=input_type
    )

    return response.embeddings
```

### 2.2 Open Source Models

| Model | Dimensions | Max Tokens | MTEB Score | Best For |
|-------|------------|------------|------------|----------|
| **BGE-large-en-v1.5** | 1024 | 512 | 64.23 | General English |
| **BGE-M3** | 1024 | 8192 | 62.0 | Multilingual, long context |
| **E5-large-v2** | 1024 | 512 | 62.68 | General purpose |
| **E5-mistral-7b** | 4096 | 32768 | 66.63 | Highest quality |
| **GTE-large** | 1024 | 512 | 63.13 | General purpose |
| **multilingual-e5-large** | 1024 | 512 | 61.50 | 100+ languages |
| **nomic-embed-text-v1.5** | 768 | 8192 | 62.28 | Long context |
| **mxbai-embed-large-v1** | 1024 | 512 | 64.68 | High quality |
| **all-MiniLM-L6-v2** | 384 | 256 | 56.26 | Fast, lightweight |
| **jina-embeddings-v2** | 768 | 8192 | 60.39 | Long context |

**Open Source Implementation:**

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads on first use)
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_local(texts: list[str]) -> list[list[float]]:
    """Embed texts using local model."""

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # For cosine similarity
    )

    return embeddings.tolist()

# BGE models need special prefixes
def embed_bge(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed using BGE with proper prefixes."""

    if is_query:
        # Add prefix for queries
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
```

### 2.3 Specialized Models

| Use Case | Recommended Model | Notes |
|----------|-------------------|-------|
| **Code** | voyage-code-2, CodeBERT | Trained on code |
| **Legal** | legal-bert-base | Legal terminology |
| **Medical** | PubMedBERT, BioBERT | Medical/scientific |
| **Financial** | FinBERT | Financial domain |
| **Multilingual** | multilingual-e5-large | 100+ languages |
| **Long Context** | nomic-embed, BGE-M3 | 8K+ tokens |

[↑ Back to Top](#table-of-contents)

---

## 3. Model Selection Guide

### 3.1 Decision Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Embedding Model Selection Guide                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Is data privacy critical?                                          │
│  ├─► Yes → Open source models (BGE, E5, GTE)                       │
│  └─► No → Continue                                                  │
│                                                                     │
│  Do you need multilingual support?                                  │
│  ├─► Yes → multilingual-e5-large or Cohere multilingual            │
│  └─► No → Continue                                                  │
│                                                                     │
│  Is your content primarily code?                                    │
│  ├─► Yes → voyage-code-2 or CodeBERT                               │
│  └─► No → Continue                                                  │
│                                                                     │
│  Do you need long context (>512 tokens)?                           │
│  ├─► Yes → nomic-embed, BGE-M3, or text-embedding-3                │
│  └─► No → Continue                                                  │
│                                                                     │
│  Is cost a primary concern?                                         │
│  ├─► Yes → text-embedding-3-small or local models                  │
│  └─► No → text-embedding-3-large or voyage-large-2                 │
│                                                                     │
│  Need highest quality?                                              │
│  ├─► Yes → E5-mistral-7b or voyage-large-2                         │
│  └─► No → BGE-large or text-embedding-3-small                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Use Case Recommendations

| Use Case | Primary Choice | Budget Alternative |
|----------|----------------|-------------------|
| **General Q&A** | text-embedding-3-small | BGE-large-en-v1.5 |
| **Enterprise Search** | text-embedding-3-large | mxbai-embed-large |
| **Customer Support** | text-embedding-3-small | all-MiniLM-L6-v2 |
| **Code Search** | voyage-code-2 | CodeBERT |
| **Legal Documents** | voyage-large-2 | legal-bert-base |
| **Multilingual** | embed-multilingual-v3.0 | multilingual-e5-large |
| **Long Documents** | text-embedding-3-large | nomic-embed-text |
| **High Volume** | text-embedding-3-small | all-MiniLM-L6-v2 |

### 3.3 Benchmarking Your Data

```python
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class EmbeddingBenchmark:
    """Benchmark embedding models on your data."""

    def __init__(self, test_data: list[dict]):
        """
        test_data format:
        [
            {"query": "...", "relevant_docs": ["doc1", "doc2"]},
            ...
        ]
        """
        self.test_data = test_data

    def evaluate_model(self, model_name: str) -> dict:
        """Evaluate a model on test data."""

        model = SentenceTransformer(model_name)

        metrics = {
            "mrr": [],      # Mean Reciprocal Rank
            "recall@5": [], # Recall at 5
            "precision@5": []
        }

        for item in self.test_data:
            query = item["query"]
            relevant = set(item["relevant_docs"])
            all_docs = item.get("all_docs", item["relevant_docs"])

            # Embed query and docs
            query_emb = model.encode(query)
            doc_embs = model.encode(all_docs)

            # Calculate similarities
            similarities = util.cos_sim(query_emb, doc_embs)[0]

            # Rank documents
            ranked_indices = similarities.argsort(descending=True)
            ranked_docs = [all_docs[i] for i in ranked_indices]

            # Calculate metrics
            metrics["mrr"].append(self.calc_mrr(ranked_docs, relevant))
            metrics["recall@5"].append(self.calc_recall_at_k(ranked_docs[:5], relevant))
            metrics["precision@5"].append(self.calc_precision_at_k(ranked_docs[:5], relevant))

        return {
            "mrr": sum(metrics["mrr"]) / len(metrics["mrr"]),
            "recall@5": sum(metrics["recall@5"]) / len(metrics["recall@5"]),
            "precision@5": sum(metrics["precision@5"]) / len(metrics["precision@5"]),
        }

    def calc_mrr(self, ranked: list, relevant: set) -> float:
        for i, doc in enumerate(ranked):
            if doc in relevant:
                return 1 / (i + 1)
        return 0

    def calc_recall_at_k(self, top_k: list, relevant: set) -> float:
        if not relevant:
            return 0
        return len(set(top_k) & relevant) / len(relevant)

    def calc_precision_at_k(self, top_k: list, relevant: set) -> float:
        if not top_k:
            return 0
        return len(set(top_k) & relevant) / len(top_k)

    def compare_models(self, model_names: list[str]) -> dict:
        """Compare multiple models."""
        results = {}
        for model_name in model_names:
            print(f"Evaluating {model_name}...")
            results[model_name] = self.evaluate_model(model_name)
        return results
```

[↑ Back to Top](#table-of-contents)

---

## 4. Fine-tuning Embeddings

### 4.1 When to Fine-tune

| Scenario | Fine-tune? | Alternative |
|----------|------------|-------------|
| General content, good results | No | Use as-is |
| Domain jargon | Maybe | Glossary preprocessing |
| Poor retrieval quality | Yes | Try different model first |
| Specialized vocabulary | Yes | - |
| Language not well supported | Yes | - |

### 4.2 Fine-tuning Approaches

#### Contrastive Learning

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def finetune_contrastive(
    base_model: str,
    train_data: list[dict],
    output_path: str,
    epochs: int = 3
):
    """
    Fine-tune using contrastive learning.

    train_data format:
    [
        {"query": "...", "positive": "relevant doc", "negative": "irrelevant doc"},
        ...
    ]
    """

    model = SentenceTransformer(base_model)

    # Prepare training examples
    train_examples = []
    for item in train_data:
        # Positive pair
        train_examples.append(InputExample(
            texts=[item["query"], item["positive"]],
            label=1.0
        ))
        # Negative pair
        if "negative" in item:
            train_examples.append(InputExample(
                texts=[item["query"], item["negative"]],
                label=0.0
            ))

    # Create dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Define loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_path
    )

    return model
```

#### Multiple Negatives Ranking Loss

```python
from sentence_transformers import losses

def finetune_mnrl(
    base_model: str,
    train_data: list[dict],
    output_path: str
):
    """
    Fine-tune using Multiple Negatives Ranking Loss.
    More efficient - uses in-batch negatives.

    train_data format:
    [
        {"query": "...", "positive": "relevant doc"},
        ...
    ]
    """

    model = SentenceTransformer(base_model)

    # Prepare examples (only need query-positive pairs)
    train_examples = [
        InputExample(texts=[item["query"], item["positive"]])
        for item in train_data
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    # MNRL uses in-batch negatives automatically
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path=output_path
    )

    return model
```

### 4.3 Creating Training Data

```python
class TrainingDataGenerator:
    """Generate training data for fine-tuning."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def generate_from_documents(self, documents: list[str]) -> list[dict]:
        """Generate query-document pairs from documents."""

        training_data = []

        for doc in documents:
            # Generate questions this document answers
            questions = await self.generate_questions(doc)

            for question in questions:
                training_data.append({
                    "query": question,
                    "positive": doc
                })

        return training_data

    async def generate_questions(self, document: str, n: int = 3) -> list[str]:
        """Generate questions answered by the document."""

        prompt = f"""
        Generate {n} natural questions that this document answers.
        Make questions diverse and natural-sounding.

        Document:
        {document[:2000]}

        Questions (one per line):
        """

        response = await self.llm.generate(prompt)
        questions = [q.strip() for q in response.split("\n") if q.strip()]
        return questions[:n]

    def add_hard_negatives(
        self,
        training_data: list[dict],
        documents: list[str],
        embedding_model
    ) -> list[dict]:
        """Add hard negatives (similar but irrelevant docs)."""

        # Embed all documents
        doc_embeddings = embedding_model.encode(documents)

        enhanced_data = []

        for item in training_data:
            query_emb = embedding_model.encode(item["query"])
            positive_doc = item["positive"]

            # Find similar documents
            similarities = util.cos_sim(query_emb, doc_embeddings)[0]
            sorted_indices = similarities.argsort(descending=True)

            # Find hard negatives (similar but not the positive)
            for idx in sorted_indices[:10]:
                candidate = documents[idx]
                if candidate != positive_doc:
                    item["negative"] = candidate
                    break

            enhanced_data.append(item)

        return enhanced_data
```

### 4.4 Matryoshka Embeddings

Train embeddings that work at multiple dimensions:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss

def finetune_matryoshka(base_model: str, train_data: list[dict]):
    """Train Matryoshka embeddings for flexible dimensions."""

    model = SentenceTransformer(base_model)

    # Define target dimensions
    matryoshka_dims = [256, 512, 768, 1024]

    # Base loss
    base_loss = losses.MultipleNegativesRankingLoss(model)

    # Wrap with Matryoshka loss
    loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=matryoshka_dims)

    # Train
    train_dataloader = DataLoader(train_examples, batch_size=32)
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=3
    )

    return model

# Usage: Truncate embeddings at runtime
full_embedding = model.encode(text)  # 1024 dims
small_embedding = full_embedding[:256]  # Still works well!
```

[↑ Back to Top](#table-of-contents)

---

## 5. Problem Statements & Solutions

### Problem 1: Poor Semantic Matching

**Symptoms:**
- Synonyms not matched ("car" vs "vehicle")
- Related concepts miss each other
- Retrieval precision is low

**Root Cause:** Embedding model doesn't capture domain semantics

**Solution:**

```python
# Option 1: Try a better model
models_to_try = [
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
    "Alibaba-NLP/gte-large-en-v1.5"
]

# Option 2: Add query expansion
async def expand_query(query: str, llm) -> str:
    prompt = f"""
    Expand this search query with synonyms and related terms:
    Query: {query}

    Expanded query (single line):
    """
    expanded = await llm.generate(prompt)
    return f"{query} {expanded}"

# Option 3: Hybrid search (add BM25)
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model

        # BM25 index
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Vector embeddings
        self.embeddings = embedding_model.encode(documents)

    def search(self, query: str, alpha: float = 0.5, k: int = 10):
        # Vector search
        query_emb = self.embedding_model.encode(query)
        vector_scores = util.cos_sim(query_emb, self.embeddings)[0]

        # BM25 search
        bm25_scores = self.bm25.get_scores(query.split())

        # Normalize scores
        vector_scores = vector_scores / vector_scores.max()
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)

        # Combine
        combined = alpha * vector_scores + (1 - alpha) * bm25_scores

        # Return top-k
        top_indices = combined.argsort(descending=True)[:k]
        return [(self.documents[i], combined[i]) for i in top_indices]
```

---

### Problem 2: Domain Vocabulary Not Understood

**Symptoms:**
- Technical terms don't match
- Acronyms fail to retrieve
- Jargon is ignored

**Root Cause:** Model not trained on domain data

**Solution:**

```python
class DomainAwareEmbedder:
    """Preprocess text to help embedding model understand domain terms."""

    def __init__(self, model_name: str, domain_glossary: dict):
        self.model = SentenceTransformer(model_name)
        self.glossary = domain_glossary  # {"API": "Application Programming Interface", ...}

    def preprocess(self, text: str) -> str:
        """Expand acronyms and add context."""

        for acronym, expansion in self.glossary.items():
            # Replace acronym with "acronym (expansion)"
            text = re.sub(
                rf'\b{acronym}\b',
                f'{acronym} ({expansion})',
                text
            )

        return text

    def encode(self, texts: list[str], preprocess: bool = True):
        if preprocess:
            texts = [self.preprocess(t) for t in texts]

        return self.model.encode(texts, normalize_embeddings=True)

# Usage
glossary = {
    "API": "Application Programming Interface",
    "SDK": "Software Development Kit",
    "JWT": "JSON Web Token",
    "OAuth": "Open Authorization"
}

embedder = DomainAwareEmbedder("BAAI/bge-large-en-v1.5", glossary)
```

---

### Problem 3: Slow Embedding Generation

**Symptoms:**
- Indexing takes too long
- Query latency is high
- Costs are high for API embeddings

**Root Cause:** Inefficient embedding pipeline

**Solution:**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedEmbedder:
    """Optimized embedding generation with batching and caching."""

    def __init__(self, model_name: str, cache_size: int = 10000):
        self.model = SentenceTransformer(model_name)
        self.cache = LRUCache(maxsize=cache_size)
        self.batch_size = 32

    def embed_batch(self, texts: list[str], use_cache: bool = True) -> list:
        """Embed texts with batching and caching."""

        results = [None] * len(texts)
        to_embed = []
        to_embed_indices = []

        # Check cache
        for i, text in enumerate(texts):
            if use_cache:
                cached = self.cache.get(text)
                if cached is not None:
                    results[i] = cached
                    continue

            to_embed.append(text)
            to_embed_indices.append(i)

        # Batch embed uncached texts
        if to_embed:
            embeddings = self.model.encode(
                to_embed,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Store results and cache
            for idx, emb, text in zip(to_embed_indices, embeddings, to_embed):
                results[idx] = emb
                if use_cache:
                    self.cache.set(text, emb)

        return results

    async def embed_async(self, texts: list[str]) -> list:
        """Async embedding for API-based models."""

        # Split into batches
        batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]

        # Process batches concurrently
        tasks = [self._embed_batch_async(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        return [emb for batch in batch_results for emb in batch]
```

---

### Problem 4: Different Embeddings for Query vs Document

**Symptoms:**
- Queries embed differently than they should
- Results don't match user intent
- BGE/E5 models underperforming

**Root Cause:** Some models need different handling for queries vs documents

**Solution:**

```python
class AsymmetricEmbedder:
    """Handle models that need different query/document treatment."""

    # Model-specific instructions
    MODEL_INSTRUCTIONS = {
        "BAAI/bge-large-en-v1.5": {
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "document_prefix": ""
        },
        "intfloat/e5-large-v2": {
            "query_prefix": "query: ",
            "document_prefix": "passage: "
        },
        "intfloat/multilingual-e5-large": {
            "query_prefix": "query: ",
            "document_prefix": "passage: "
        }
    }

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.instructions = self.MODEL_INSTRUCTIONS.get(model_name, {})

    def embed_query(self, query: str):
        """Embed a search query."""
        prefix = self.instructions.get("query_prefix", "")
        return self.model.encode(prefix + query, normalize_embeddings=True)

    def embed_documents(self, documents: list[str]):
        """Embed documents for indexing."""
        prefix = self.instructions.get("document_prefix", "")
        texts = [prefix + doc for doc in documents]
        return self.model.encode(texts, normalize_embeddings=True)
```

---

### Problem 5: High API Costs

**Symptoms:**
- Embedding API bills are high
- Cost scales with document volume
- Budget constraints

**Solution:**

```python
class CostOptimizedEmbedder:
    """Minimize embedding API costs."""

    def __init__(self):
        # Use local model for documents (free)
        self.local_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Use API model for queries only (high quality, low volume)
        self.api_client = OpenAI()

        # Cache to prevent re-embedding
        self.cache = PersistentCache("embeddings.db")

    def embed_documents(self, documents: list[str]) -> list:
        """Embed documents using free local model."""
        return self.local_model.encode(documents, normalize_embeddings=True)

    def embed_query(self, query: str) -> list:
        """Embed query using API (optional, or use local)."""

        # Check cache
        cached = self.cache.get(query)
        if cached:
            return cached

        # For queries, local is usually fine
        embedding = self.local_model.encode(query, normalize_embeddings=True)

        # Cache it
        self.cache.set(query, embedding)

        return embedding

    def estimate_cost(self, num_documents: int, avg_tokens: int) -> dict:
        """Estimate embedding costs."""

        # API pricing (OpenAI text-embedding-3-small)
        api_cost_per_million = 0.02

        # Local is free
        local_cost = 0

        # Calculate
        total_tokens = num_documents * avg_tokens
        api_cost = (total_tokens / 1_000_000) * api_cost_per_million

        return {
            "api_cost": f"${api_cost:.2f}",
            "local_cost": "$0",
            "recommendation": "local" if api_cost > 1 else "either"
        }
```

[↑ Back to Top](#table-of-contents)

---

## 6. Trade-offs

### Model Type Trade-offs

| Factor | Proprietary (API) | Open Source (Local) |
|--------|-------------------|---------------------|
| **Quality** | ✅ Generally higher | ⚠️ Varies by model |
| **Cost** | ❌ Per-token pricing | ✅ Free (compute only) |
| **Privacy** | ❌ Data sent to provider | ✅ Data stays local |
| **Latency** | ⚠️ Network dependent | ✅ Local processing |
| **Scaling** | ✅ Auto-scales | ❌ Requires infrastructure |
| **Maintenance** | ✅ Provider handles | ❌ Self-managed |

### Dimension Trade-offs

| Dimensions | Quality | Storage | Search Speed |
|------------|---------|---------|--------------|
| 256 | ⭐⭐ | 1x | ⚡⚡⚡⚡⚡ |
| 512 | ⭐⭐⭐ | 2x | ⚡⚡⚡⚡ |
| 768 | ⭐⭐⭐⭐ | 3x | ⚡⚡⚡ |
| 1024 | ⭐⭐⭐⭐ | 4x | ⚡⚡ |
| 1536 | ⭐⭐⭐⭐⭐ | 6x | ⚡⚡ |
| 3072 | ⭐⭐⭐⭐⭐ | 12x | ⚡ |

### Fine-tuning Trade-offs

| Approach | Data Needed | Compute Cost | Quality Gain |
|----------|-------------|--------------|--------------|
| No fine-tuning | 0 | $0 | Baseline |
| Contrastive | 1K+ pairs | Low | 5-15% |
| MNRL | 1K+ pairs | Low | 5-15% |
| Hard negatives | 10K+ | Medium | 10-20% |
| Full fine-tune | 100K+ | High | 15-30% |

[↑ Back to Top](#table-of-contents)

---

## 7. Cost-Effective Solutions

### Cost Comparison

| Model | Cost per 1M Tokens | 10K Docs (500 tokens) | 1M Docs |
|-------|--------------------|-----------------------|---------|
| text-embedding-3-large | $0.13 | $0.65 | $65 |
| text-embedding-3-small | $0.02 | $0.10 | $10 |
| Cohere embed-v3 | $0.10 | $0.50 | $50 |
| Local (BGE/E5) | $0 | $0 + compute | $0 + compute |

### Free Stack Recommendation

```python
# Best free embedding setup for most cases

from sentence_transformers import SentenceTransformer

# Primary: High-quality general model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Alternative: Faster, smaller
fast_model = SentenceTransformer("all-MiniLM-L6-v2")

# For multilingual
multilingual_model = SentenceTransformer("intfloat/multilingual-e5-large")
```

### Hybrid Cost Strategy

```python
class HybridEmbedder:
    """Use local for bulk, API for quality-critical."""

    def __init__(self):
        self.local = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.api = OpenAI()

    def embed_documents(self, documents: list[str]):
        """Documents: use free local model."""
        return self.local.encode(documents, normalize_embeddings=True)

    def embed_query_standard(self, query: str):
        """Standard queries: use free local model."""
        return self.local.encode(query, normalize_embeddings=True)

    def embed_query_premium(self, query: str):
        """Premium/complex queries: use API."""
        response = self.api.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
```

### Caching Strategy

```python
import hashlib
import sqlite3

class EmbeddingCache:
    """Persistent cache to avoid re-embedding."""

    def __init__(self, db_path: str = "embedding_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                model TEXT,
                embedding BLOB
            )
        """)

    def get(self, text: str, model: str) -> list | None:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ? AND model = ?",
            (text_hash, model)
        )
        row = cursor.fetchone()
        if row:
            import numpy as np
            return np.frombuffer(row[0]).tolist()
        return None

    def set(self, text: str, model: str, embedding: list):
        import numpy as np
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        embedding_bytes = np.array(embedding).tobytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
            (text_hash, model, embedding_bytes)
        )
        self.conn.commit()
```

[↑ Back to Top](#table-of-contents)

---

## 8. Best Practices

### DO's

1. **Normalize Embeddings**
   ```python
   embeddings = model.encode(texts, normalize_embeddings=True)
   ```

2. **Batch Processing**
   ```python
   # Good - batch processing
   embeddings = model.encode(documents, batch_size=32)

   # Bad - one at a time
   embeddings = [model.encode(doc) for doc in documents]
   ```

3. **Use Correct Query/Document Prefixes**
   ```python
   # For BGE models
   query_emb = model.encode("Represent this sentence for searching: " + query)
   doc_emb = model.encode(document)  # No prefix
   ```

4. **Benchmark on Your Data**
   ```python
   # Don't trust generic benchmarks - test on your domain
   benchmark = EmbeddingBenchmark(your_test_data)
   results = benchmark.compare_models(["model1", "model2"])
   ```

5. **Monitor Embedding Quality**
   ```python
   # Track similarity distribution
   similarities = util.cos_sim(query_embs, doc_embs)
   avg_similarity = similarities.mean()
   # Alert if distribution shifts significantly
   ```

### DON'Ts

1. **Don't Mix Embedding Models**
   - Query and document embeddings must use same model

2. **Don't Ignore Model Limits**
   - Truncate text to max_tokens before embedding

3. **Don't Skip Normalization for Cosine Similarity**
   - Either normalize or use dot product

4. **Don't Embed Empty Strings**
   ```python
   texts = [t for t in texts if t.strip()]  # Filter empty
   ```

5. **Don't Forget Caching**
   - Cache embeddings to avoid re-computation

[↑ Back to Top](#table-of-contents)

---

## 9. Quick Reference

### Model Selection Cheat Sheet

```
Use Case → Recommended Model
─────────────────────────────────────────
Budget-friendly API   → text-embedding-3-small
Highest quality API   → text-embedding-3-large
Privacy-required      → BGE-large-en-v1.5
Multilingual          → multilingual-e5-large
Code search           → voyage-code-2
Long context          → nomic-embed-text-v1.5
Fast/lightweight      → all-MiniLM-L6-v2
```

### Implementation Checklist

```
□ Choose appropriate model for use case
□ Implement batching for efficiency
□ Add caching layer
□ Use correct query/document prefixes
□ Normalize embeddings
□ Handle max token limits
□ Benchmark on domain data
□ Monitor quality in production
```

### Cost Quick Reference

```
Embedding Costs (per 1M tokens):
─────────────────────────────────────────
text-embedding-3-small:  $0.02
text-embedding-3-large:  $0.13
Cohere embed-v3:         $0.10
Local models:            $0 + compute
```

### Code Template

```python
from sentence_transformers import SentenceTransformer

class ProductionEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.cache = {}

    def embed_query(self, query: str):
        return self.model.encode(
            f"Represent this sentence for searching: {query}",
            normalize_embeddings=True
        )

    def embed_documents(self, documents: list[str]):
        return self.model.encode(
            documents,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Chunking](./02-chunking.md) | [Main Guide](../README.md) | [Vector Databases →](./04-vector-databases.md) |

---

*Last updated: 2024*
