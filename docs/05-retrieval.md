# Retrieval Strategies for RAG

Complete guide to finding and ranking relevant content for generation.

[← Back to Main Guide](../README.md) | [← Previous: Vector Databases](./04-vector-databases.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Retrieval Methods](#2-retrieval-methods)
- [3. Re-ranking](#3-re-ranking)
- [4. Query Processing](#4-query-processing)
- [5. Advanced Retrieval Patterns](#5-advanced-retrieval-patterns)
- [6. Problem Statements & Solutions](#6-problem-statements--solutions)
- [7. Trade-offs](#7-trade-offs)
- [8. Cost-Effective Solutions](#8-cost-effective-solutions)
- [9. Best Practices](#9-best-practices)
- [10. Quick Reference](#10-quick-reference)

---

## 1. Overview

### What is Retrieval in RAG?

Retrieval is the process of finding relevant documents from your knowledge base to provide context for the LLM.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Query         Query Processing        Initial Retrieval      │
│  ──────────         ────────────────        ─────────────────      │
│  "How do I          ┌─────────────┐        ┌─────────────┐        │
│   reset my    ───▶  │  Expand /   │  ───▶  │   Vector    │        │
│   password?"        │  Rewrite    │        │   Search    │        │
│                     └─────────────┘        └──────┬──────┘        │
│                                                   │                 │
│                                                   ▼                 │
│  Final Context      Post-Processing         Re-ranking             │
│  ─────────────      ───────────────         ──────────             │
│  ┌─────────────┐    ┌─────────────┐        ┌─────────────┐        │
│  │  Top 3-5    │◀───│  Dedupe /   │◀───────│   Cross-    │        │
│  │  Documents  │    │  Filter     │        │   Encoder   │        │
│  └─────────────┘    └─────────────┘        └─────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Precision@K** | % of retrieved docs that are relevant | > 80% |
| **Recall@K** | % of relevant docs that are retrieved | > 70% |
| **MRR** | Mean Reciprocal Rank of first relevant result | > 0.6 |
| **NDCG** | Normalized Discounted Cumulative Gain | > 0.7 |
| **Latency** | Time to retrieve results | < 100ms |

### Retrieval Quality Impact

```
Retrieval Quality → RAG Output Quality
─────────────────────────────────────────

Poor Retrieval           Good Retrieval
────────────             ──────────────
• Irrelevant context  →  • Relevant context
• Hallucinations      →  • Grounded answers
• Missing info        →  • Complete answers
• User frustration    →  • User satisfaction
```

[↑ Back to Top](#table-of-contents)

---

## 2. Retrieval Methods

### 2.1 Dense Retrieval (Vector Search)

Uses embeddings to find semantically similar documents.

```python
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """Retrieve using vector similarity."""

    def __init__(self, embedding_model: str, vector_store):
        self.model = SentenceTransformer(embedding_model)
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 10) -> list:
        # Embed query
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k
        )

        return results
```

| Pros | Cons |
|------|------|
| Semantic understanding | May miss exact keyword matches |
| Handles paraphrasing | Requires good embeddings |
| Language-agnostic | Computationally expensive |
| Good for natural language | Can miss rare terms |

**When to Use:**
- Natural language queries
- Conceptual/semantic search
- When exact keywords may vary

---

### 2.2 Sparse Retrieval (Keyword Search)

Uses term frequency and inverse document frequency (TF-IDF, BM25).

```python
from rank_bm25 import BM25Okapi
import numpy as np

class SparseRetriever:
    """Retrieve using BM25 keyword matching."""

    def __init__(self, documents: list[str]):
        # Tokenize documents
        self.documents = documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int = 10) -> list:
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {"document": self.documents[i], "score": scores[i]}
            for i in top_indices
        ]
```

| Pros | Cons |
|------|------|
| Exact keyword matching | No semantic understanding |
| Fast and efficient | Vocabulary mismatch issues |
| Interpretable | Language/domain specific |
| Great for known terms | Misses paraphrases |

**When to Use:**
- Keyword-heavy queries
- Technical terms, codes, IDs
- When exact matches matter

---

### 2.3 Hybrid Search

Combines dense and sparse retrieval for best of both worlds.

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridRetriever:
    """Combine vector and keyword search."""

    def __init__(self, documents: list[str], embedding_model: str):
        self.documents = documents

        # Dense retriever
        self.encoder = SentenceTransformer(embedding_model)
        self.doc_embeddings = self.encoder.encode(
            documents, normalize_embeddings=True
        )

        # Sparse retriever
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5  # Weight for vector search
    ) -> list:
        # Dense scores
        query_emb = self.encoder.encode(query, normalize_embeddings=True)
        dense_scores = np.dot(self.doc_embeddings, query_emb)

        # Sparse scores
        sparse_scores = self.bm25.get_scores(query.lower().split())

        # Normalize scores to [0, 1]
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-6)

        # Combine scores
        combined = alpha * dense_norm + (1 - alpha) * sparse_norm

        # Get top-k
        top_indices = np.argsort(combined)[::-1][:top_k]

        return [
            {
                "document": self.documents[i],
                "score": combined[i],
                "dense_score": dense_scores[i],
                "sparse_score": sparse_scores[i]
            }
            for i in top_indices
        ]
```

**Reciprocal Rank Fusion (RRF):**

```python
def reciprocal_rank_fusion(
    result_lists: list[list],
    k: int = 60
) -> list:
    """
    Combine multiple ranked lists using RRF.

    RRF score = Σ 1 / (k + rank)
    """
    scores = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            doc_id = item["id"]
            if doc_id not in scores:
                scores[doc_id] = {"item": item, "score": 0}
            scores[doc_id]["score"] += 1 / (k + rank + 1)

    # Sort by combined score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    return [item["item"] for item in ranked]

# Usage
dense_results = dense_retriever.retrieve(query, top_k=20)
sparse_results = sparse_retriever.retrieve(query, top_k=20)
combined = reciprocal_rank_fusion([dense_results, sparse_results])
```

| Pros | Cons |
|------|------|
| Best of both worlds | More complex |
| Handles diverse queries | Requires tuning alpha |
| Higher recall | Slightly slower |
| More robust | Two systems to maintain |

**When to Use:**
- Production systems
- Diverse query types
- When maximum recall matters

---

### 2.4 Retrieval Methods Comparison

| Method | Semantic | Keywords | Speed | Complexity |
|--------|----------|----------|-------|------------|
| Dense only | ✅ | ❌ | Fast | Low |
| Sparse only | ❌ | ✅ | Very Fast | Low |
| Hybrid | ✅ | ✅ | Medium | Medium |
| Hybrid + Rerank | ✅ | ✅ | Slower | High |

[↑ Back to Top](#table-of-contents)

---

## 3. Re-ranking

### 3.1 Why Re-rank?

```
Initial Retrieval (Bi-encoder)        Re-ranking (Cross-encoder)
──────────────────────────────        ──────────────────────────

Fast but approximate:                 Slow but accurate:
┌─────────────────────────┐          ┌─────────────────────────┐
│ Query ──▶ Embedding     │          │ [Query, Doc1] ──▶ Score │
│                         │          │ [Query, Doc2] ──▶ Score │
│ Doc ──▶ Embedding       │          │ [Query, Doc3] ──▶ Score │
│                         │          │         ...             │
│ Score = similarity      │          │                         │
└─────────────────────────┘          └─────────────────────────┘

Use case: Search millions           Use case: Re-rank top 20-50
Speed: ~1ms for 1M docs             Speed: ~50ms for 20 docs
```

### 3.2 Cross-Encoder Re-ranking

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Re-rank results using cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list:
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Score pairs
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [
            {"document": doc, "score": score}
            for doc, score in scored_docs[:top_k]
        ]

# Usage
retriever = HybridRetriever(documents, "BAAI/bge-large-en-v1.5")
reranker = CrossEncoderReranker()

# Initial retrieval (fast, get candidates)
candidates = retriever.retrieve(query, top_k=20)

# Re-rank (slower, improve precision)
reranked = reranker.rerank(
    query,
    [c["document"] for c in candidates],
    top_k=5
)
```

### 3.3 Re-ranking Models

| Model | Speed | Quality | Size |
|-------|-------|---------|------|
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Fast | Good | 22M |
| **cross-encoder/ms-marco-MiniLM-L-12-v2** | Medium | Better | 33M |
| **BAAI/bge-reranker-base** | Medium | Better | 278M |
| **BAAI/bge-reranker-large** | Slow | Best | 560M |
| **Cohere Rerank** | API | Excellent | - |
| **Voyage Rerank** | API | Excellent | - |

### 3.4 Cohere Rerank (API)

```python
import cohere

co = cohere.Client("your-api-key")

def rerank_with_cohere(query: str, documents: list[str], top_k: int = 5):
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k
    )

    return [
        {
            "document": documents[r.index],
            "score": r.relevance_score
        }
        for r in response.results
    ]
```

### 3.5 LLM-Based Re-ranking

```python
class LLMReranker:
    """Use LLM to re-rank documents."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list:
        # Format documents with indices
        doc_list = "\n".join([
            f"[{i}] {doc[:500]}..."
            for i, doc in enumerate(documents)
        ])

        prompt = f"""
        Given the query and documents below, rank the documents by relevance.
        Return only the indices of the top {top_k} most relevant documents, comma-separated.

        Query: {query}

        Documents:
        {doc_list}

        Most relevant document indices (comma-separated):
        """

        response = await self.llm.generate(prompt)

        # Parse indices
        try:
            indices = [int(i.strip()) for i in response.split(",")]
            return [documents[i] for i in indices[:top_k] if i < len(documents)]
        except:
            return documents[:top_k]  # Fallback
```

[↑ Back to Top](#table-of-contents)

---

## 4. Query Processing

### 4.1 Query Expansion

Add related terms to improve recall.

```python
class QueryExpander:
    """Expand queries with related terms."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def expand(self, query: str) -> str:
        prompt = f"""
        Expand this search query with synonyms and related terms.
        Keep it as a single search query, not a list.

        Original query: {query}

        Expanded query:
        """

        expanded = await self.llm.generate(prompt)
        return f"{query} {expanded}"

    def expand_with_synonyms(self, query: str, synonym_dict: dict) -> str:
        """Expand using predefined synonyms."""
        expanded_terms = []
        for word in query.split():
            expanded_terms.append(word)
            if word.lower() in synonym_dict:
                expanded_terms.extend(synonym_dict[word.lower()])

        return " ".join(expanded_terms)

# Usage
expander = QueryExpander(llm)
original = "How do I reset my password?"
expanded = await expander.expand(original)
# "How do I reset my password? change credentials login authentication forgot"
```

### 4.2 Query Decomposition

Break complex queries into sub-queries.

```python
class QueryDecomposer:
    """Decompose complex queries into simpler sub-queries."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def decompose(self, query: str) -> list[str]:
        prompt = f"""
        Break this complex question into simpler, independent sub-questions.
        Each sub-question should be answerable on its own.

        Complex question: {query}

        Sub-questions (one per line):
        """

        response = await self.llm.generate(prompt)
        sub_queries = [q.strip() for q in response.split("\n") if q.strip()]
        return sub_queries

    async def retrieve_and_combine(
        self,
        query: str,
        retriever,
        top_k: int = 5
    ) -> list:
        # Decompose
        sub_queries = await self.decompose(query)

        # Retrieve for each sub-query
        all_results = []
        for sub_query in sub_queries:
            results = retriever.retrieve(sub_query, top_k=top_k)
            all_results.extend(results)

        # Deduplicate and combine
        seen = set()
        unique_results = []
        for result in all_results:
            doc_id = hash(result["document"])
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(result)

        return unique_results

# Usage
decomposer = QueryDecomposer(llm)

query = "Compare the pricing and features of AWS S3 vs Google Cloud Storage for enterprise use"
sub_queries = await decomposer.decompose(query)
# [
#   "What is AWS S3 pricing for enterprise?",
#   "What are AWS S3 features for enterprise?",
#   "What is Google Cloud Storage pricing for enterprise?",
#   "What are Google Cloud Storage features for enterprise?"
# ]
```

### 4.3 Query Rewriting

Improve query quality for better retrieval.

```python
class QueryRewriter:
    """Rewrite queries for better retrieval."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def rewrite(self, query: str, context: str = None) -> str:
        """Rewrite query to be more specific and searchable."""

        prompt = f"""
        Rewrite this query to be more specific and searchable.
        Make it clear and unambiguous.
        {f"Previous context: {context}" if context else ""}

        Original query: {query}

        Improved query:
        """

        return await self.llm.generate(prompt)

    async def make_standalone(self, query: str, chat_history: list) -> str:
        """Convert follow-up question to standalone query."""

        history_str = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in chat_history[-3:]  # Last 3 turns
        ])

        prompt = f"""
        Given the conversation history, rewrite the follow-up question
        as a standalone question that captures the full context.

        Conversation:
        {history_str}

        Follow-up question: {query}

        Standalone question:
        """

        return await self.llm.generate(prompt)

# Usage
rewriter = QueryRewriter(llm)

# Simple rewrite
query = "how to fix it"
better = await rewriter.rewrite(query, context="discussing login errors")
# "How to fix login authentication errors"

# Standalone from conversation
history = [
    {"user": "Tell me about AWS S3", "assistant": "AWS S3 is..."},
    {"user": "What about pricing?", "assistant": "S3 pricing..."}
]
query = "How does it compare to Google?"
standalone = await rewriter.make_standalone(query, history)
# "How does AWS S3 pricing compare to Google Cloud Storage?"
```

### 4.4 HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer to improve retrieval.

```python
class HyDERetriever:
    """Hypothetical Document Embeddings retrieval."""

    def __init__(self, llm_client, embedding_model, vector_store):
        self.llm = llm_client
        self.encoder = embedding_model
        self.vector_store = vector_store

    async def retrieve(self, query: str, top_k: int = 5) -> list:
        # Generate hypothetical answer
        prompt = f"""
        Write a detailed paragraph that would answer this question:
        {query}

        Answer:
        """

        hypothetical_doc = await self.llm.generate(prompt)

        # Embed the hypothetical document
        hyde_embedding = self.encoder.encode(
            hypothetical_doc,
            normalize_embeddings=True
        )

        # Search with hypothetical embedding
        results = self.vector_store.search(
            query_vector=hyde_embedding,
            limit=top_k
        )

        return results

# Usage
hyde = HyDERetriever(llm, encoder, vector_store)

query = "What are the side effects of aspirin?"
# HyDE generates: "Aspirin may cause stomach irritation, bleeding..."
# Then searches using that embedding
results = await hyde.retrieve(query)
```

[↑ Back to Top](#table-of-contents)

---

## 5. Advanced Retrieval Patterns

### 5.1 Parent Document Retrieval

Retrieve small chunks but return larger context.

```python
class ParentDocumentRetriever:
    """Retrieve by child, return parent for context."""

    def __init__(
        self,
        vector_store,       # Stores child chunks
        document_store,     # Stores parent documents
        child_to_parent     # Mapping of child_id -> parent_id
    ):
        self.vector_store = vector_store
        self.document_store = document_store
        self.child_to_parent = child_to_parent

    def retrieve(self, query_embedding, top_k: int = 5) -> list:
        # Search child chunks (precise retrieval)
        child_results = self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k * 2  # Get more children
        )

        # Map to parents (richer context)
        parent_ids = set()
        for child in child_results:
            parent_id = self.child_to_parent.get(child["id"])
            if parent_id:
                parent_ids.add(parent_id)

        # Get parent documents
        parents = []
        for parent_id in list(parent_ids)[:top_k]:
            parent = self.document_store.get(parent_id)
            parents.append(parent)

        return parents
```

### 5.2 Multi-Query Retrieval

Generate multiple perspectives on the query.

```python
class MultiQueryRetriever:
    """Generate multiple query variations for better coverage."""

    def __init__(self, llm_client, retriever):
        self.llm = llm_client
        self.retriever = retriever

    async def retrieve(self, query: str, top_k: int = 5) -> list:
        # Generate query variations
        variations = await self.generate_variations(query)

        # Retrieve for each variation
        all_results = []
        for variation in variations:
            results = self.retriever.retrieve(variation, top_k=top_k)
            all_results.extend(results)

        # Deduplicate and combine using RRF
        combined = reciprocal_rank_fusion(
            [all_results[i:i+top_k] for i in range(0, len(all_results), top_k)]
        )

        return combined[:top_k]

    async def generate_variations(self, query: str, n: int = 3) -> list[str]:
        prompt = f"""
        Generate {n} different ways to ask this question.
        Each variation should approach the topic from a different angle.

        Original question: {query}

        Variations (one per line):
        """

        response = await self.llm.generate(prompt)
        variations = [query]  # Include original
        variations.extend([v.strip() for v in response.split("\n") if v.strip()])

        return variations[:n+1]
```

### 5.3 Self-Query Retrieval

Use LLM to extract metadata filters from natural language.

```python
class SelfQueryRetriever:
    """Extract filters from natural language queries."""

    def __init__(self, llm_client, vector_store, metadata_schema: dict):
        self.llm = llm_client
        self.vector_store = vector_store
        self.metadata_schema = metadata_schema

    async def retrieve(self, query: str, top_k: int = 5) -> list:
        # Extract filter and search query
        parsed = await self.parse_query(query)

        # Apply filter and search
        results = self.vector_store.search(
            query_vector=self.encode(parsed["search_query"]),
            filter=parsed["filter"],
            limit=top_k
        )

        return results

    async def parse_query(self, query: str) -> dict:
        schema_desc = "\n".join([
            f"- {field}: {info['description']} (type: {info['type']})"
            for field, info in self.metadata_schema.items()
        ])

        prompt = f"""
        Extract search query and metadata filters from this question.

        Available metadata fields:
        {schema_desc}

        Question: {query}

        Return JSON with:
        - search_query: the semantic search query
        - filter: metadata filter conditions (or null if none)

        JSON:
        """

        response = await self.llm.generate(prompt)
        return json.loads(response)

# Usage
schema = {
    "category": {"description": "Document category", "type": "string"},
    "date": {"description": "Document date", "type": "date"},
    "author": {"description": "Document author", "type": "string"}
}

retriever = SelfQueryRetriever(llm, vector_store, schema)

query = "Find security policies written by John after 2024"
# Extracts:
# - search_query: "security policies"
# - filter: {"author": "John", "date": {"$gte": "2024-01-01"}}
```

### 5.4 Contextual Compression

Compress retrieved documents to relevant portions.

```python
class ContextualCompressor:
    """Extract only relevant portions from retrieved documents."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def compress(
        self,
        query: str,
        documents: list[str],
        max_length: int = 500
    ) -> list[str]:
        compressed = []

        for doc in documents:
            relevant = await self.extract_relevant(query, doc, max_length)
            if relevant:
                compressed.append(relevant)

        return compressed

    async def extract_relevant(
        self,
        query: str,
        document: str,
        max_length: int
    ) -> str:
        prompt = f"""
        Extract only the portions of this document that are relevant to the query.
        If nothing is relevant, return "NOT_RELEVANT".
        Keep the extraction under {max_length} characters.

        Query: {query}

        Document:
        {document}

        Relevant extraction:
        """

        response = await self.llm.generate(prompt)

        if "NOT_RELEVANT" in response:
            return None
        return response[:max_length]
```

### 5.5 Ensemble Retrieval Pipeline

Combine everything into a production pipeline.

```python
class EnsembleRetrievalPipeline:
    """Production retrieval pipeline with all optimizations."""

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        reranker,
        query_processor,
        compressor=None
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker
        self.query_processor = query_processor
        self.compressor = compressor

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_expansion: bool = True,
        use_rerank: bool = True,
        use_compression: bool = False
    ) -> list:
        # Step 1: Process query
        processed_query = query
        if use_expansion:
            processed_query = await self.query_processor.expand(query)

        # Step 2: Hybrid retrieval
        dense_results = self.dense.retrieve(processed_query, top_k=20)
        sparse_results = self.sparse.retrieve(processed_query, top_k=20)

        # Step 3: Combine with RRF
        combined = reciprocal_rank_fusion([dense_results, sparse_results])
        candidates = combined[:20]

        # Step 4: Re-rank
        if use_rerank:
            reranked = self.reranker.rerank(
                query,
                [c["document"] for c in candidates],
                top_k=top_k
            )
        else:
            reranked = candidates[:top_k]

        # Step 5: Compress (optional)
        if use_compression and self.compressor:
            documents = [r["document"] for r in reranked]
            compressed = await self.compressor.compress(query, documents)
            return compressed

        return reranked
```

[↑ Back to Top](#table-of-contents)

---

## 6. Problem Statements & Solutions

### Problem 1: Missing Keyword Matches

**Symptoms:**
- Exact term queries return no results
- Product codes, IDs not found
- Technical terms missed

**Root Cause:** Pure vector search doesn't do exact matching

**Solution:**

```python
# Implement hybrid search
class KeywordAwareRetriever:
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        # Build BM25 index
        self.bm25 = BM25Okapi([doc.split() for doc in documents])
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 10):
        # Check if query contains potential keywords/codes
        has_keywords = self.detect_keywords(query)

        if has_keywords:
            # Weight BM25 higher
            alpha = 0.3  # 30% vector, 70% keyword
        else:
            # Weight vector higher
            alpha = 0.7

        # Hybrid search
        return self.hybrid_search(query, alpha, top_k)

    def detect_keywords(self, query: str) -> bool:
        # Detect product codes, IDs, etc.
        import re
        patterns = [
            r'\b[A-Z]{2,}\d+\b',  # Product codes like "SKU123"
            r'\b\d{4,}\b',         # Long numbers
            r'\b[A-Z]{3,}\b'       # Acronyms
        ]
        return any(re.search(p, query) for p in patterns)
```

---

### Problem 2: Poor Results for Complex Questions

**Symptoms:**
- Multi-part questions get incomplete answers
- Comparison questions miss one side
- "How does X compare to Y" fails

**Root Cause:** Single retrieval can't address multiple aspects

**Solution:**

```python
class ComplexQueryHandler:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    async def retrieve(self, query: str, top_k: int = 5):
        # Detect query type
        query_type = await self.classify_query(query)

        if query_type == "comparison":
            return await self.handle_comparison(query, top_k)
        elif query_type == "multi_part":
            return await self.handle_multi_part(query, top_k)
        else:
            return self.retriever.retrieve(query, top_k)

    async def handle_comparison(self, query: str, top_k: int):
        # Extract entities being compared
        entities = await self.extract_entities(query)

        all_results = []
        for entity in entities:
            # Retrieve for each entity
            sub_query = f"{entity} {query}"
            results = self.retriever.retrieve(sub_query, top_k=top_k)
            all_results.extend(results)

        # Ensure balanced representation
        return self.balance_results(all_results, entities, top_k)

    async def handle_multi_part(self, query: str, top_k: int):
        # Decompose into sub-queries
        sub_queries = await self.decompose(query)

        all_results = []
        for sub_q in sub_queries:
            results = self.retriever.retrieve(sub_q, top_k=top_k//len(sub_queries))
            all_results.extend(results)

        return self.deduplicate(all_results)[:top_k]
```

---

### Problem 3: Hallucination Due to Poor Retrieval

**Symptoms:**
- LLM makes up information
- Answers not grounded in documents
- Confident but wrong responses

**Root Cause:** Retrieved context is irrelevant or insufficient

**Solution:**

```python
class GroundedRetriever:
    """Ensure retrieved content is relevant before generation."""

    def __init__(self, retriever, relevance_checker):
        self.retriever = retriever
        self.relevance_checker = relevance_checker

    async def retrieve(self, query: str, top_k: int = 5, min_relevance: float = 0.5):
        # Get candidates
        candidates = self.retriever.retrieve(query, top_k=top_k * 2)

        # Filter by relevance
        relevant = []
        for candidate in candidates:
            relevance = await self.relevance_checker.check(query, candidate["document"])

            if relevance >= min_relevance:
                candidate["relevance"] = relevance
                relevant.append(candidate)

        # Sort by relevance
        relevant.sort(key=lambda x: x["relevance"], reverse=True)

        # If not enough relevant docs, return with warning
        if len(relevant) < top_k // 2:
            return {
                "documents": relevant,
                "warning": "Low confidence - few relevant documents found"
            }

        return {"documents": relevant[:top_k], "warning": None}

class RelevanceChecker:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def check(self, query: str, document: str) -> float:
        prompt = f"""
        Rate how relevant this document is to the query on a scale of 0-1.
        0 = completely irrelevant
        1 = highly relevant

        Query: {query}
        Document: {document[:1000]}

        Relevance score (just the number):
        """

        response = await self.llm.generate(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
```

---

### Problem 4: Slow Retrieval Performance

**Symptoms:**
- Queries take >500ms
- Can't scale to production load
- Re-ranking is bottleneck

**Root Cause:** Unoptimized pipeline

**Solution:**

```python
import asyncio
from functools import lru_cache

class OptimizedRetriever:
    """Optimized retrieval with caching and parallelization."""

    def __init__(self, dense_retriever, sparse_retriever, reranker):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker
        self.cache = {}

    async def retrieve(self, query: str, top_k: int = 5):
        # Check cache
        cache_key = f"{query}:{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Parallel retrieval
        dense_task = asyncio.create_task(
            self.async_dense(query, top_k * 2)
        )
        sparse_task = asyncio.create_task(
            self.async_sparse(query, top_k * 2)
        )

        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )

        # Combine
        combined = reciprocal_rank_fusion([dense_results, sparse_results])

        # Lightweight re-ranking (only if needed)
        if len(combined) > top_k:
            reranked = self.reranker.rerank(
                query,
                [c["document"] for c in combined[:15]],  # Limit candidates
                top_k=top_k
            )
        else:
            reranked = combined[:top_k]

        # Cache result
        self.cache[cache_key] = reranked

        return reranked

    @lru_cache(maxsize=1000)
    def cached_embed(self, text: str):
        return self.dense.encode(text)
```

---

### Problem 5: Context Window Overflow

**Symptoms:**
- Too many tokens for LLM
- Truncated context
- Missing important information

**Root Cause:** Retrieved too much content

**Solution:**

```python
class ContextWindowManager:
    """Manage context to fit within LLM limits."""

    def __init__(self, max_tokens: int = 4000, tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer or self.default_tokenizer

    def fit_context(
        self,
        query: str,
        documents: list[dict],
        reserve_for_response: int = 500
    ) -> list[dict]:
        available_tokens = self.max_tokens - reserve_for_response - self.count_tokens(query)

        selected = []
        used_tokens = 0

        for doc in documents:
            doc_tokens = self.count_tokens(doc["content"])

            if used_tokens + doc_tokens <= available_tokens:
                selected.append(doc)
                used_tokens += doc_tokens
            else:
                # Try to fit truncated version
                remaining = available_tokens - used_tokens
                if remaining > 100:  # Worth including partial
                    truncated = self.truncate(doc["content"], remaining)
                    doc["content"] = truncated
                    doc["truncated"] = True
                    selected.append(doc)
                break

        return selected

    def strategic_ordering(self, documents: list) -> list:
        """Order documents to avoid 'lost in the middle' problem."""
        if len(documents) <= 2:
            return documents

        # Put most relevant at beginning and end
        n = len(documents)
        ordered = []

        for i in range(n):
            if i % 2 == 0:
                ordered.insert(0, documents[i])
            else:
                ordered.append(documents[i])

        return ordered

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text)[:max_tokens]
        return self.tokenizer.decode(tokens)
```

[↑ Back to Top](#table-of-contents)

---

## 7. Trade-offs

### Retrieval Method Trade-offs

| Method | Precision | Recall | Speed | Complexity |
|--------|-----------|--------|-------|------------|
| Dense only | Medium | High | Fast | Low |
| Sparse only | High | Low | Very Fast | Low |
| Hybrid | High | High | Medium | Medium |
| + Re-ranking | Very High | High | Slow | High |
| + Query expansion | High | Very High | Slow | High |

### Re-ranking Trade-offs

| Approach | Quality | Speed | Cost |
|----------|---------|-------|------|
| No re-ranking | Baseline | Fast | Free |
| Cross-encoder (small) | +10-15% | ~50ms | Free |
| Cross-encoder (large) | +15-25% | ~200ms | Free |
| Cohere Rerank | +20-30% | ~100ms | $2/1K |
| LLM re-ranking | +25-35% | ~500ms | $$$ |

### Query Processing Trade-offs

| Technique | Recall Gain | Latency | Cost |
|-----------|-------------|---------|------|
| Query expansion | +10-20% | +100ms | $ |
| Query decomposition | +15-25% | +200ms | $ |
| HyDE | +20-30% | +300ms | $$ |
| Multi-query | +15-25% | +400ms | $$ |

[↑ Back to Top](#table-of-contents)

---

## 8. Cost-Effective Solutions

### Free Retrieval Stack

```python
# Complete free retrieval setup

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Embedding model (free, local)
encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Re-ranker (free, local)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# BM25 (free)
bm25 = BM25Okapi(tokenized_docs)

# Vector store (free)
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
```

### Cost Comparison

| Component | Free Option | Paid Option | Paid Cost |
|-----------|-------------|-------------|-----------|
| Embeddings | BGE, E5 (local) | OpenAI | $0.02/1M tokens |
| Vector DB | Chroma, pgvector | Pinecone | ~$70/month |
| Re-ranking | Cross-encoder | Cohere | $2/1K queries |
| Query expansion | Local LLM | GPT-4 | ~$0.01/query |

### Optimization Tips

1. **Cache Everything**
   ```python
   # Cache embeddings
   @lru_cache(maxsize=10000)
   def embed(text):
       return encoder.encode(text)

   # Cache search results
   @lru_cache(maxsize=1000)
   def search(query):
       return retriever.retrieve(query)
   ```

2. **Use Smaller Models for Re-ranking**
   ```python
   # MiniLM is 10x faster than base models
   reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   ```

3. **Limit Re-ranking Candidates**
   ```python
   # Re-rank top 15 instead of top 50
   candidates = retriever.retrieve(query, top_k=15)
   reranked = reranker.rerank(query, candidates, top_k=5)
   ```

4. **Skip Re-ranking for Simple Queries**
   ```python
   if is_simple_query(query):
       return retriever.retrieve(query, top_k=5)
   else:
       return rerank(retriever.retrieve(query, top_k=20))
   ```

[↑ Back to Top](#table-of-contents)

---

## 9. Best Practices

### DO's

1. **Start with Hybrid Search**
   ```python
   # Default: 60% vector, 40% keyword
   combined = 0.6 * dense_score + 0.4 * sparse_score
   ```

2. **Always Re-rank for Quality-Critical Apps**
   ```python
   candidates = retriever.retrieve(query, top_k=20)
   final = reranker.rerank(query, candidates, top_k=5)
   ```

3. **Use Metadata Filters**
   ```python
   results = vector_store.search(
       query_vector=embedding,
       filter={"category": "technical", "date": {"$gte": "2024-01-01"}}
   )
   ```

4. **Monitor Retrieval Quality**
   ```python
   metrics = {
       "precision@5": calculate_precision(results, relevant),
       "recall@10": calculate_recall(results, relevant),
       "mrr": calculate_mrr(results, relevant)
   }
   ```

5. **Handle No Results Gracefully**
   ```python
   results = retriever.retrieve(query)
   if not results or max(r["score"] for r in results) < 0.5:
       return "I couldn't find relevant information. Could you rephrase?"
   ```

### DON'Ts

1. **Don't Use Vector Search Alone for Keywords**
   - Add BM25 for exact matches

2. **Don't Skip Relevance Checks**
   - Low-relevance results cause hallucinations

3. **Don't Over-Retrieve**
   - More context isn't always better

4. **Don't Ignore Query Quality**
   - Process and improve queries before search

5. **Don't Forget About Latency**
   - Users expect <1 second responses

[↑ Back to Top](#table-of-contents)

---

## 10. Quick Reference

### Retrieval Strategy Cheat Sheet

```
Query Type → Strategy
─────────────────────────────────────────
Keyword-heavy     → Hybrid (BM25 weighted)
Semantic          → Dense + Re-ranking
Complex/Multi-hop → Query decomposition
Comparison        → Multi-entity retrieval
Conversational    → Query rewriting
```

### Re-ranking Model Cheat Sheet

```
Scenario → Model
─────────────────────────────────────────
Fast + Good       → ms-marco-MiniLM-L-6-v2
Best Free         → bge-reranker-large
Best Quality      → Cohere Rerank API
Code              → Custom trained
```

### Retrieval Pipeline Template

```python
# Production-ready retrieval pipeline

class ProductionRetriever:
    def __init__(self):
        # Dense retrieval
        self.encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Sparse retrieval
        self.bm25 = BM25Okapi(tokenized_docs)

        # Re-ranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query: str, top_k: int = 5):
        # 1. Hybrid search
        dense = self.dense_search(query, k=20)
        sparse = self.sparse_search(query, k=20)
        combined = self.rrf_fusion([dense, sparse])

        # 2. Re-rank
        reranked = self.reranker.rerank(query, combined[:15])

        # 3. Return top-k
        return reranked[:top_k]
```

### Key Metrics Targets

```
Metric          Minimum    Good      Excellent
───────────────────────────────────────────────
Precision@5     > 0.6     > 0.8     > 0.9
Recall@10       > 0.5     > 0.7     > 0.85
MRR             > 0.5     > 0.7     > 0.85
Latency (p95)   < 500ms   < 200ms   < 100ms
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Vector Databases](./04-vector-databases.md) | [Main Guide](../README.md) | - |

---

*Last updated: 2024*
