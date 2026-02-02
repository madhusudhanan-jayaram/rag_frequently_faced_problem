# Generation & Prompting for RAG

Complete guide to constructing effective prompts and generating high-quality responses.

[← Back to Main Guide](../README.md) | [← Previous: Retrieval](./05-retrieval.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Prompt Engineering Fundamentals](#2-prompt-engineering-fundamentals)
- [3. RAG Prompt Patterns](#3-rag-prompt-patterns)
- [4. Context Window Management](#4-context-window-management)
- [5. Response Generation Strategies](#5-response-generation-strategies)
- [6. Hallucination Prevention](#6-hallucination-prevention)
- [7. Problem Statements & Solutions](#7-problem-statements--solutions)
- [8. Trade-offs](#8-trade-offs)
- [9. Cost-Effective Solutions](#9-cost-effective-solutions)
- [10. Best Practices](#10-best-practices)
- [11. Quick Reference](#11-quick-reference)

---

## 1. Overview

### The Generation Phase

Generation is where retrieved context meets user query to produce the final response.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Retrieved          Prompt              LLM              Response   │
│  Context            Assembly            Generation       Processing │
│  ────────           ────────            ──────────       ────────── │
│  ┌─────────┐       ┌─────────┐        ┌─────────┐      ┌─────────┐│
│  │ Doc 1   │       │ System  │        │         │      │ Validate││
│  │ Doc 2   │  ───▶ │ Context │  ───▶  │  LLM    │ ───▶ │ Format  ││
│  │ Doc 3   │       │ Query   │        │         │      │ Return  ││
│  └─────────┘       └─────────┘        └─────────┘      └─────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Grounding** | Answers must be based on retrieved context |
| **Attribution** | Cite sources for claims |
| **Honesty** | Acknowledge when information is missing |
| **Relevance** | Stay focused on the user's question |
| **Clarity** | Provide clear, actionable responses |

[↑ Back to Top](#table-of-contents)

---

## 2. Prompt Engineering Fundamentals

### 2.1 Prompt Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG Prompt Structure                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ SYSTEM PROMPT                                                │   │
│  │ - Role definition                                            │   │
│  │ - Behavior guidelines                                        │   │
│  │ - Output format instructions                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ CONTEXT                                                      │   │
│  │ - Retrieved documents                                        │   │
│  │ - Source metadata                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ USER QUERY                                                   │   │
│  │ - Original question                                          │   │
│  │ - Any clarifications                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT INSTRUCTIONS                                          │   │
│  │ - Format requirements                                        │   │
│  │ - Citation format                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Basic RAG Prompt

```python
BASIC_RAG_PROMPT = """
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Answer:
"""
```

### 2.3 Structured RAG Prompt

```python
STRUCTURED_RAG_PROMPT = """
You are an expert assistant for {company_name}. Your role is to provide accurate,
helpful answers based on the company's documentation.

## Instructions
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information"
3. Cite sources using [Source: document_name]
4. Be concise but complete

## Context
{context}

## Question
{question}

## Answer Format
- Start with a direct answer
- Provide supporting details
- Include citations
- Note any limitations

Answer:
"""
```

### 2.4 System Prompts by Use Case

```python
SYSTEM_PROMPTS = {
    "customer_support": """
        You are a friendly customer support agent for {company}.
        - Be empathetic and helpful
        - Provide step-by-step solutions when applicable
        - Escalate to human support for complex issues
        - Never make promises about refunds or compensation
    """,

    "technical_docs": """
        You are a technical documentation assistant.
        - Provide precise, accurate technical information
        - Include code examples when relevant
        - Reference specific documentation sections
        - Warn about deprecated features or breaking changes
    """,

    "legal_assistant": """
        You are a legal information assistant.
        - Provide information, not legal advice
        - Always recommend consulting a qualified attorney
        - Cite specific clauses and sections
        - Note jurisdictional limitations
    """,

    "research_assistant": """
        You are a research assistant.
        - Synthesize information from multiple sources
        - Highlight consensus and disagreements
        - Note the recency and reliability of sources
        - Suggest areas for further research
    """
}
```

[↑ Back to Top](#table-of-contents)

---

## 3. RAG Prompt Patterns

### 3.1 Citation-Required Prompt

```python
CITATION_PROMPT = """
Answer the question using the numbered sources below.
You MUST cite sources using [1], [2], etc. for every claim.

Sources:
{numbered_sources}

Question: {question}

Instructions:
- Every factual claim must have a citation
- Use multiple citations if information comes from multiple sources
- If sources conflict, note the discrepancy
- If no source supports an answer, say "Not found in provided sources"

Answer with citations:
"""

def format_sources(documents: list) -> str:
    return "\n\n".join([
        f"[{i+1}] {doc['title']}\n{doc['content']}"
        for i, doc in enumerate(documents)
    ])
```

### 3.2 Step-by-Step Reasoning Prompt

```python
REASONING_PROMPT = """
Answer the question by reasoning through the context step by step.

Context:
{context}

Question: {question}

Think through this step by step:
1. What information in the context is relevant to this question?
2. What can we conclude from this information?
3. Are there any gaps or uncertainties?

Reasoning:
[Your step-by-step analysis]

Final Answer:
[Your concise answer based on the reasoning]
"""
```

### 3.3 Comparison Prompt

```python
COMPARISON_PROMPT = """
Compare the items based on the provided context.

Context:
{context}

Items to compare: {items}
Comparison criteria: {criteria}

Provide your comparison in this format:

## Summary
[Brief overview of key differences]

## Detailed Comparison

| Criteria | {item1} | {item2} |
|----------|---------|---------|
| [criteria1] | [value] | [value] |
| [criteria2] | [value] | [value] |

## Recommendation
[If applicable, which is better for what use case]

Sources: [cite relevant sources]
"""
```

### 3.4 Multi-Turn Conversation Prompt

```python
CONVERSATION_PROMPT = """
You are a helpful assistant engaged in a conversation.

## Conversation History
{history}

## Relevant Context for Current Question
{context}

## Current Question
{question}

## Instructions
- Consider the conversation history for context
- Answer the current question based on the provided context
- Reference previous discussion when relevant
- Ask clarifying questions if needed

Response:
"""

def format_history(turns: list) -> str:
    return "\n".join([
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in turns[-5:]  # Last 5 turns
    ])
```

### 3.5 Structured Output Prompt

```python
STRUCTURED_OUTPUT_PROMPT = """
Extract information from the context and return as structured JSON.

Context:
{context}

Query: {query}

Return a JSON object with this structure:
{{
    "answer": "direct answer to the query",
    "confidence": "high/medium/low",
    "sources": ["list of source documents used"],
    "key_facts": ["list of relevant facts extracted"],
    "related_topics": ["topics for further exploration"],
    "limitations": "any caveats or missing information"
}}

JSON Response:
"""
```

### 3.6 Refusal Prompt (When Context Doesn't Help)

```python
HONEST_PROMPT = """
Answer based ONLY on the provided context.

Context:
{context}

Question: {question}

IMPORTANT RULES:
1. If the context contains the answer → Provide it with citations
2. If the context partially answers → Share what you found and note what's missing
3. If the context is irrelevant → Say "I don't have information about this in the provided documents"

NEVER make up information. NEVER use knowledge outside the context.

Response:
"""
```

[↑ Back to Top](#table-of-contents)

---

## 4. Context Window Management

### 4.1 Token Budget Planning

```python
class TokenBudgetManager:
    """Manage token allocation across prompt components."""

    def __init__(self, model_max_tokens: int = 8192):
        self.max_tokens = model_max_tokens

    def calculate_budget(
        self,
        system_prompt: str,
        query: str,
        max_response: int = 1000
    ) -> dict:
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        reserved = system_tokens + query_tokens + max_response

        available_for_context = self.max_tokens - reserved

        return {
            "total_budget": self.max_tokens,
            "system_prompt": system_tokens,
            "query": query_tokens,
            "response_reserve": max_response,
            "context_budget": available_for_context,
            "buffer": 100  # Safety margin
        }

    def fit_context(
        self,
        documents: list[dict],
        budget: int
    ) -> list[dict]:
        """Fit documents within token budget."""
        selected = []
        used = 0

        for doc in documents:
            doc_tokens = self.count_tokens(doc["content"])

            if used + doc_tokens <= budget - 100:  # Buffer
                selected.append(doc)
                used += doc_tokens
            else:
                # Try to fit truncated
                remaining = budget - used - 100
                if remaining > 200:
                    truncated = self.truncate_to_tokens(doc["content"], remaining)
                    doc["content"] = truncated
                    doc["truncated"] = True
                    selected.append(doc)
                break

        return selected

    def count_tokens(self, text: str) -> int:
        # Use tiktoken for accurate count
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)[:max_tokens]
        return enc.decode(tokens) + "..."
```

### 4.2 Context Ordering Strategies

```python
class ContextOrderer:
    """Order context to maximize LLM attention."""

    def order_for_attention(self, documents: list[dict]) -> list[dict]:
        """
        Address 'lost in the middle' problem.
        LLMs pay more attention to beginning and end.
        """
        if len(documents) <= 2:
            return documents

        # Most relevant at beginning and end
        n = len(documents)
        ordered = []

        # Alternate: first goes to start, second to end, etc.
        for i, doc in enumerate(documents):
            if i % 2 == 0:
                ordered.insert(0, doc)
            else:
                ordered.append(doc)

        return ordered

    def order_by_relevance(self, documents: list[dict]) -> list[dict]:
        """Simply order by relevance score."""
        return sorted(documents, key=lambda x: x.get("score", 0), reverse=True)

    def order_chronologically(self, documents: list[dict]) -> list[dict]:
        """Order by document date for temporal questions."""
        return sorted(documents, key=lambda x: x.get("date", ""), reverse=True)
```

### 4.3 Context Compression

```python
class ContextCompressor:
    """Compress context to fit more information."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def compress(
        self,
        documents: list[dict],
        query: str,
        target_tokens: int
    ) -> str:
        """Compress documents to target token count."""

        # Method 1: Extractive compression
        extractive = await self.extractive_compress(documents, query)

        if self.count_tokens(extractive) <= target_tokens:
            return extractive

        # Method 2: Abstractive compression
        return await self.abstractive_compress(extractive, query, target_tokens)

    async def extractive_compress(
        self,
        documents: list[dict],
        query: str
    ) -> str:
        """Extract only relevant sentences."""
        prompt = f"""
        Extract only the sentences that are relevant to answering this question.
        Remove all irrelevant information.

        Question: {query}

        Documents:
        {self.format_docs(documents)}

        Relevant excerpts only:
        """

        return await self.llm.generate(prompt)

    async def abstractive_compress(
        self,
        text: str,
        query: str,
        target_tokens: int
    ) -> str:
        """Summarize while preserving key information."""
        prompt = f"""
        Summarize this information in approximately {target_tokens} tokens.
        Preserve all facts relevant to the question.

        Question: {query}

        Information:
        {text}

        Compressed summary:
        """

        return await self.llm.generate(prompt)
```

### 4.4 Dynamic Context Selection

```python
class DynamicContextSelector:
    """Dynamically select context based on query complexity."""

    def __init__(self, retriever, budget_manager):
        self.retriever = retriever
        self.budget = budget_manager

    async def select_context(
        self,
        query: str,
        query_complexity: str
    ) -> list[dict]:
        """Select context amount based on query complexity."""

        configs = {
            "simple": {"k": 3, "max_tokens_per_doc": 300},
            "moderate": {"k": 5, "max_tokens_per_doc": 500},
            "complex": {"k": 8, "max_tokens_per_doc": 800},
            "comprehensive": {"k": 12, "max_tokens_per_doc": 1000}
        }

        config = configs.get(query_complexity, configs["moderate"])

        # Retrieve
        documents = self.retriever.retrieve(query, top_k=config["k"])

        # Truncate each document
        for doc in documents:
            if self.budget.count_tokens(doc["content"]) > config["max_tokens_per_doc"]:
                doc["content"] = self.budget.truncate_to_tokens(
                    doc["content"],
                    config["max_tokens_per_doc"]
                )

        return documents
```

[↑ Back to Top](#table-of-contents)

---

## 5. Response Generation Strategies

### 5.1 Stuff (Simple Concatenation)

```python
class StuffStrategy:
    """Simplest approach: concatenate all context."""

    async def generate(
        self,
        llm,
        query: str,
        documents: list[dict],
        system_prompt: str
    ) -> str:
        context = "\n\n---\n\n".join([
            f"Source: {doc.get('source', 'Unknown')}\n{doc['content']}"
            for doc in documents
        ])

        prompt = f"""
        {system_prompt}

        Context:
        {context}

        Question: {query}

        Answer:
        """

        return await llm.generate(prompt)
```

**When to use:** Context fits in window, simple questions

### 5.2 Map-Reduce

```python
class MapReduceStrategy:
    """Process documents individually, then combine."""

    async def generate(
        self,
        llm,
        query: str,
        documents: list[dict]
    ) -> str:
        # Map: Extract relevant info from each document
        summaries = []
        for doc in documents:
            summary = await self.map_document(llm, query, doc)
            if summary:
                summaries.append(summary)

        # Reduce: Combine summaries into final answer
        return await self.reduce_summaries(llm, query, summaries)

    async def map_document(self, llm, query: str, doc: dict) -> str:
        prompt = f"""
        Extract information relevant to this question from the document.
        If nothing is relevant, respond with "NO_RELEVANT_INFO".

        Question: {query}

        Document:
        {doc['content']}

        Relevant information:
        """

        response = await llm.generate(prompt)
        return None if "NO_RELEVANT_INFO" in response else response

    async def reduce_summaries(self, llm, query: str, summaries: list) -> str:
        combined = "\n\n".join(summaries)

        prompt = f"""
        Synthesize these extracted pieces of information to answer the question.

        Question: {query}

        Extracted information:
        {combined}

        Comprehensive answer:
        """

        return await llm.generate(prompt)
```

**When to use:** Many documents, need comprehensive analysis

### 5.3 Refine

```python
class RefineStrategy:
    """Iteratively refine answer with each document."""

    async def generate(
        self,
        llm,
        query: str,
        documents: list[dict]
    ) -> str:
        # Start with first document
        answer = await self.initial_answer(llm, query, documents[0])

        # Refine with remaining documents
        for doc in documents[1:]:
            answer = await self.refine_answer(llm, query, doc, answer)

        return answer

    async def initial_answer(self, llm, query: str, doc: dict) -> str:
        prompt = f"""
        Answer this question based on the document.

        Question: {query}

        Document:
        {doc['content']}

        Answer:
        """

        return await llm.generate(prompt)

    async def refine_answer(
        self,
        llm,
        query: str,
        doc: dict,
        current_answer: str
    ) -> str:
        prompt = f"""
        Refine the existing answer using the new document.
        Only update if the new document provides additional relevant information.

        Question: {query}

        Current answer:
        {current_answer}

        New document:
        {doc['content']}

        Refined answer:
        """

        return await llm.generate(prompt)
```

**When to use:** Need detailed synthesis, documents build on each other

### 5.4 Streaming Responses

```python
class StreamingGenerator:
    """Stream responses for better UX."""

    async def generate_stream(
        self,
        llm,
        prompt: str
    ):
        """Yield response chunks as they're generated."""
        async for chunk in llm.stream(prompt):
            yield chunk

    async def generate_with_sources(
        self,
        llm,
        query: str,
        documents: list[dict]
    ):
        """Stream answer, then append sources."""

        prompt = self.build_prompt(query, documents)

        # Stream the answer
        answer_parts = []
        async for chunk in llm.stream(prompt):
            answer_parts.append(chunk)
            yield chunk

        # Append sources at the end
        sources = self.format_sources(documents)
        yield f"\n\n---\nSources:\n{sources}"
```

[↑ Back to Top](#table-of-contents)

---

## 6. Hallucination Prevention

### 6.1 Types of Hallucination

| Type | Description | Prevention |
|------|-------------|------------|
| **Intrinsic** | Uses training knowledge instead of context | Strong grounding instructions |
| **Extrinsic** | Fabricates facts not in context | Citation requirements |
| **Extrapolation** | Over-generalizes from context | Constrain to explicit info |
| **Conflation** | Mixes information incorrectly | Source isolation |

### 6.2 Grounding Techniques

```python
GROUNDED_PROMPT = """
CRITICAL: You must ONLY use information from the provided context.

Context:
{context}

Question: {question}

Rules:
1. If the answer is in the context → Provide it and cite the source
2. If the answer is NOT in the context → Say "This information is not available in the provided documents"
3. NEVER use prior knowledge or make assumptions
4. NEVER extrapolate beyond what's explicitly stated
5. If uncertain → Express uncertainty explicitly

Answer (cite sources for every claim):
"""
```

### 6.3 Citation Enforcement

```python
class CitationEnforcer:
    """Ensure responses include proper citations."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def generate_with_citations(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        # Number the sources
        numbered_docs = self.number_documents(documents)

        prompt = f"""
        Answer using ONLY the numbered sources below.
        EVERY sentence must end with a citation like [1] or [1,2].

        Sources:
        {numbered_docs}

        Question: {query}

        Answer (with citations after every sentence):
        """

        response = await self.llm.generate(prompt)

        # Validate citations
        validation = self.validate_citations(response, len(documents))

        return {
            "answer": response,
            "citations_valid": validation["valid"],
            "uncited_sentences": validation["uncited"],
            "invalid_citations": validation["invalid"]
        }

    def validate_citations(self, response: str, num_sources: int) -> dict:
        import re

        sentences = re.split(r'[.!?]+', response)
        citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'

        uncited = []
        invalid = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            citations = re.findall(citation_pattern, sentence)

            if not citations:
                uncited.append(sentence)
            else:
                for cite_group in citations:
                    for cite in cite_group.split(','):
                        cite_num = int(cite.strip())
                        if cite_num < 1 or cite_num > num_sources:
                            invalid.append(cite_num)

        return {
            "valid": len(uncited) == 0 and len(invalid) == 0,
            "uncited": uncited,
            "invalid": list(set(invalid))
        }
```

### 6.4 Fact Verification Layer

```python
class FactVerifier:
    """Verify generated claims against source documents."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def verify_response(
        self,
        response: str,
        context: str
    ) -> dict:
        # Extract claims from response
        claims = await self.extract_claims(response)

        # Verify each claim
        verified = []
        unverified = []

        for claim in claims:
            is_supported = await self.verify_claim(claim, context)
            if is_supported:
                verified.append(claim)
            else:
                unverified.append(claim)

        return {
            "verified_claims": verified,
            "unverified_claims": unverified,
            "verification_rate": len(verified) / len(claims) if claims else 1.0,
            "is_grounded": len(unverified) == 0
        }

    async def extract_claims(self, response: str) -> list[str]:
        prompt = f"""
        Extract all factual claims from this response.
        List each claim on a new line.

        Response:
        {response}

        Claims (one per line):
        """

        result = await self.llm.generate(prompt)
        return [c.strip() for c in result.split('\n') if c.strip()]

    async def verify_claim(self, claim: str, context: str) -> bool:
        prompt = f"""
        Determine if this claim is supported by the context.

        Claim: {claim}

        Context:
        {context}

        Is this claim directly supported by the context? (yes/no)
        """

        result = await self.llm.generate(prompt)
        return 'yes' in result.lower()
```

### 6.5 Confidence Scoring

```python
class ConfidenceScorer:
    """Score confidence in generated response."""

    async def score_response(
        self,
        llm,
        query: str,
        response: str,
        documents: list[dict]
    ) -> dict:
        prompt = f"""
        Evaluate the confidence level of this response.

        Question: {query}

        Response: {response}

        Source documents: {len(documents)} documents provided

        Rate on these dimensions (1-5):
        1. Evidence support: How well is the response supported by sources?
        2. Completeness: Does it fully answer the question?
        3. Clarity: Is the response clear and unambiguous?
        4. Relevance: Does it directly address the question?

        Return scores as JSON:
        {{"evidence": X, "completeness": X, "clarity": X, "relevance": X, "overall": X}}
        """

        scores = await llm.generate(prompt)

        return {
            "scores": json.loads(scores),
            "confidence_level": self.calculate_confidence(json.loads(scores))
        }

    def calculate_confidence(self, scores: dict) -> str:
        avg = sum(scores.values()) / len(scores)
        if avg >= 4:
            return "high"
        elif avg >= 3:
            return "medium"
        else:
            return "low"
```

[↑ Back to Top](#table-of-contents)

---

## 7. Problem Statements & Solutions

### Problem 1: LLM Ignores Context

**Symptoms:**
- Answers based on training data, not documents
- Context seems ignored
- Generic responses

**Solution:**

```python
CONTEXT_ENFORCEMENT_PROMPT = """
IMPORTANT: This is a retrieval-augmented system. You MUST:
1. Base your answer ONLY on the context below
2. If the context doesn't contain the answer, say "Not found in provided documents"
3. NEVER use your training knowledge to answer

To prove you're using the context, you must:
- Quote relevant passages directly
- Cite source documents
- Mention specific details from the context

Context:
{context}

Question: {question}

Answer (quoting from context):
"""

# Add context validation
class ContextAwareGenerator:
    async def generate(self, query: str, context: str) -> dict:
        response = await self.llm.generate(
            CONTEXT_ENFORCEMENT_PROMPT.format(context=context, question=query)
        )

        # Check if response mentions context specifics
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        overlap = len(context_words & response_words) / len(context_words)

        if overlap < 0.1:  # Low overlap might indicate ignoring context
            # Regenerate with stronger prompt
            response = await self.regenerate_with_emphasis(query, context)

        return {"response": response, "context_usage": overlap}
```

---

### Problem 2: Verbose, Unfocused Responses

**Symptoms:**
- Long, rambling answers
- Includes irrelevant information
- Hard to find the actual answer

**Solution:**

```python
CONCISE_PROMPT = """
Answer the question directly and concisely.

Context:
{context}

Question: {question}

Instructions:
- Start with the direct answer in the first sentence
- Keep total response under 150 words
- Only include information that directly answers the question
- Use bullet points for multiple items
- End with sources

Direct Answer:
"""

# Post-processing for length
class ConciseGenerator:
    MAX_WORDS = 150

    async def generate(self, query: str, context: str) -> str:
        response = await self.llm.generate(
            CONCISE_PROMPT.format(context=context, question=query)
        )

        # If too long, summarize
        if len(response.split()) > self.MAX_WORDS:
            response = await self.summarize(response, query)

        return response

    async def summarize(self, response: str, query: str) -> str:
        prompt = f"""
        Summarize this response to under {self.MAX_WORDS} words.
        Keep the direct answer and key facts.

        Original: {response}

        Question being answered: {query}

        Concise version:
        """
        return await self.llm.generate(prompt)
```

---

### Problem 3: Missing Citations

**Symptoms:**
- No source attribution
- Can't verify claims
- Compliance issues

**Solution:**

```python
class MandatoryCitationGenerator:
    """Force citation in every claim."""

    CITATION_PROMPT = """
    Answer with mandatory citations for every factual claim.

    Sources:
    {sources}

    Question: {question}

    FORMAT REQUIREMENT:
    Every sentence containing a fact must end with [Source N].
    Example: "The policy allows 30-day returns [Source 1]."

    If you cannot cite a source, do not include that information.

    Answer:
    """

    async def generate(self, query: str, documents: list) -> dict:
        sources = self.format_sources(documents)

        response = await self.llm.generate(
            self.CITATION_PROMPT.format(sources=sources, question=query)
        )

        # Validate all sentences have citations
        uncited = self.find_uncited_sentences(response)

        if uncited:
            # Request re-generation for uncited parts
            response = await self.add_missing_citations(response, uncited, documents)

        return {
            "response": response,
            "citation_count": self.count_citations(response),
            "all_cited": len(uncited) == 0
        }
```

---

### Problem 4: "I Don't Know" Even When Answer Exists

**Symptoms:**
- Unnecessarily says "not found"
- Misses obvious answers
- Too conservative

**Solution:**

```python
THOROUGH_SEARCH_PROMPT = """
Carefully search the context for information related to the question.

Context:
{context}

Question: {question}

Before answering "I don't know", check for:
1. Direct answers to the question
2. Partial information that could help
3. Related information that provides context
4. Synonyms or rephrased versions of the query terms

If you find ANY relevant information, share it.
Only say "not found" if absolutely nothing is relevant.

Analysis of context relevance:
[First, analyze what relevant info exists]

Answer:
[Based on your analysis]
"""

# Two-pass approach
class ThoroughGenerator:
    async def generate(self, query: str, context: str) -> str:
        # First pass: check for any relevance
        relevance_check = await self.check_relevance(query, context)

        if relevance_check["has_relevant_info"]:
            return await self.generate_answer(query, context)
        else:
            return self.not_found_response(relevance_check["closest_topics"])

    async def check_relevance(self, query: str, context: str) -> dict:
        prompt = f"""
        Does this context contain any information related to the question?

        Question: {query}
        Context: {context[:2000]}

        Return JSON:
        {{
            "has_relevant_info": true/false,
            "relevant_sections": ["quotes from relevant parts"],
            "closest_topics": ["topics in context that might be related"]
        }}
        """
        return json.loads(await self.llm.generate(prompt))
```

---

### Problem 5: Inconsistent Response Format

**Symptoms:**
- Sometimes markdown, sometimes plain text
- Varying structures
- Hard to parse programmatically

**Solution:**

```python
STRUCTURED_FORMAT_PROMPT = """
Answer in this exact format:

## Summary
[1-2 sentence direct answer]

## Details
[Bullet points with supporting information]
- Point 1 [Source]
- Point 2 [Source]

## Sources
- Source 1: [title]
- Source 2: [title]

---

Context: {context}

Question: {question}

Response (follow the format exactly):
"""

class FormattedGenerator:
    REQUIRED_SECTIONS = ["## Summary", "## Details", "## Sources"]

    async def generate(self, query: str, context: str) -> dict:
        response = await self.llm.generate(
            STRUCTURED_FORMAT_PROMPT.format(context=context, question=query)
        )

        # Validate format
        missing = [s for s in self.REQUIRED_SECTIONS if s not in response]

        if missing:
            response = await self.fix_format(response, missing)

        return {
            "response": response,
            "format_valid": len(missing) == 0,
            "sections": self.parse_sections(response)
        }
```

[↑ Back to Top](#table-of-contents)

---

## 8. Trade-offs

### Generation Strategy Trade-offs

| Strategy | Quality | Speed | Cost | Best For |
|----------|---------|-------|------|----------|
| Stuff | Medium | Fast | $ | Short context |
| Map-Reduce | High | Slow | $$$ | Many documents |
| Refine | Very High | Very Slow | $$$$ | Detailed synthesis |
| Streaming | Medium | Perceived fast | $ | UX |

### Prompt Complexity Trade-offs

| Approach | Control | Flexibility | Token Cost |
|----------|---------|-------------|------------|
| Simple prompt | Low | High | Low |
| Structured prompt | High | Medium | Medium |
| Few-shot examples | Very High | Low | High |

### Context Size Trade-offs

| Context Size | Answer Quality | Latency | Cost |
|--------------|----------------|---------|------|
| Minimal (1-2 docs) | May miss info | Fast | Low |
| Moderate (3-5 docs) | Balanced | Medium | Medium |
| Large (10+ docs) | Comprehensive | Slow | High |

[↑ Back to Top](#table-of-contents)

---

## 9. Cost-Effective Solutions

### Prompt Optimization

```python
# Reduce tokens while maintaining quality

# Instead of verbose:
verbose = """
You are a helpful, knowledgeable assistant who always provides
accurate, well-researched answers based on the provided context...
"""

# Use concise:
concise = """
Answer from context only. Cite sources. Say "unknown" if not found.
"""

# Token savings: ~50%
```

### Model Selection by Task

| Task | Recommended Model | Cost |
|------|-------------------|------|
| Simple Q&A | GPT-3.5-turbo | $ |
| Complex reasoning | GPT-4-turbo | $$$ |
| Structured extraction | GPT-3.5 + validation | $ |
| Summarization | Claude Haiku | $ |

### Caching Strategies

```python
class ResponseCache:
    """Cache responses for common queries."""

    def __init__(self):
        self.exact_cache = {}
        self.semantic_cache = SemanticCache()

    async def get_or_generate(
        self,
        query: str,
        context_hash: str,
        generate_fn
    ) -> str:
        # Check exact cache
        cache_key = f"{query}:{context_hash}"
        if cache_key in self.exact_cache:
            return self.exact_cache[cache_key]

        # Check semantic cache
        similar = await self.semantic_cache.find_similar(query)
        if similar:
            return similar

        # Generate new response
        response = await generate_fn()

        # Cache it
        self.exact_cache[cache_key] = response
        await self.semantic_cache.add(query, response)

        return response
```

[↑ Back to Top](#table-of-contents)

---

## 10. Best Practices

### DO's

1. **Always Require Citations**
   ```python
   prompt += "\nCite sources for every factual claim using [Source N]."
   ```

2. **Set Clear Boundaries**
   ```python
   prompt += "\nOnly use the provided context. Never use external knowledge."
   ```

3. **Handle Uncertainty Explicitly**
   ```python
   prompt += "\nIf uncertain, say 'Based on the available information...' "
   ```

4. **Use Consistent Formatting**
   ```python
   prompt += "\nFormat: Summary → Details → Sources"
   ```

5. **Validate Outputs**
   ```python
   response = await generate(prompt)
   validated = await verify_citations(response, documents)
   ```

### DON'Ts

1. **Don't Trust LLM Blindly**
   - Always verify critical claims

2. **Don't Overload Context**
   - More isn't always better

3. **Don't Skip Error Handling**
   ```python
   if not context:
       return "I need documents to answer this question."
   ```

4. **Don't Ignore User Intent**
   - Match response style to question type

5. **Don't Forget Edge Cases**
   - Empty context, ambiguous queries, multiple valid answers

[↑ Back to Top](#table-of-contents)

---

## 11. Quick Reference

### Prompt Template Cheat Sheet

```python
# Basic RAG
"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

# With citations
"Sources:\n{sources}\n\nQuestion: {question}\n\nAnswer with [N] citations:"

# Grounded
"Answer ONLY from context. Say 'not found' if missing.\n\nContext: {context}\n\n..."

# Structured output
"Return JSON: {schema}\n\nContext: {context}\n\nQuery: {query}"
```

### Token Budget Guidelines

```
Component          Typical %    Example (8K model)
──────────────────────────────────────────────────
System prompt      5-10%        400-800 tokens
Context            50-70%       4000-5600 tokens
Query              2-5%         160-400 tokens
Response reserve   15-25%       1200-2000 tokens
Buffer             5%           400 tokens
```

### Quality Checklist

```
□ Response uses context (not training data)
□ All claims have citations
□ Uncertainty is acknowledged
□ Format is consistent
□ Length is appropriate
□ No hallucinated facts
□ Sources are valid
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Retrieval](./05-retrieval.md) | [Main Guide](../README.md) | [Evaluation & Monitoring →](./07-evaluation-monitoring.md) |

---

*Last updated: 2024*
