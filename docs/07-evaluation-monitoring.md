# Evaluation & Monitoring for RAG

Complete guide to measuring, tracking, and improving RAG system quality.

[← Back to Main Guide](../README.md) | [← Previous: Generation & Prompting](./06-generation-prompting.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Evaluation Metrics](#2-evaluation-metrics)
- [3. Evaluation Frameworks](#3-evaluation-frameworks)
- [4. Testing Strategies](#4-testing-strategies)
- [5. Production Monitoring](#5-production-monitoring)
- [6. Problem Statements & Solutions](#6-problem-statements--solutions)
- [7. Trade-offs](#7-trade-offs)
- [8. Cost-Effective Solutions](#8-cost-effective-solutions)
- [9. Best Practices](#9-best-practices)
- [10. Quick Reference](#10-quick-reference)

---

## 1. Overview

### Why Evaluation Matters

```
┌─────────────────────────────────────────────────────────────────────┐
│                Without Evaluation vs With Evaluation                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Without                              With                          │
│  ───────                              ────                          │
│  • "It seems to work"                • "87% retrieval precision"   │
│  • Unknown failure modes             • Known weak areas            │
│  • Can't compare changes             • Data-driven decisions       │
│  • Surprise production issues        • Proactive fixes             │
│  • No accountability                 • Clear metrics               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Evaluation Dimensions

| Dimension | What It Measures | Key Metrics |
|-----------|------------------|-------------|
| **Retrieval Quality** | Finding relevant documents | Precision, Recall, MRR |
| **Generation Quality** | Answer correctness | Faithfulness, Relevance |
| **End-to-End Quality** | Overall system performance | User satisfaction, Task completion |
| **Performance** | Speed and efficiency | Latency, Throughput |
| **Cost** | Resource usage | Cost per query |

[↑ Back to Top](#table-of-contents)

---

## 2. Evaluation Metrics

### 2.1 Retrieval Metrics

#### Precision@K

```python
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    What fraction of retrieved documents are relevant?

    Precision@K = |Retrieved ∩ Relevant| / K
    """
    retrieved_k = retrieved[:k]
    relevant_count = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_count / k

# Example
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc7"}
print(precision_at_k(retrieved, relevant, 5))  # 0.4 (2 out of 5)
```

#### Recall@K

```python
def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    What fraction of relevant documents did we find?

    Recall@K = |Retrieved ∩ Relevant| / |Relevant|
    """
    retrieved_k = set(retrieved[:k])
    found = len(retrieved_k & relevant)
    return found / len(relevant) if relevant else 0

# Example
print(recall_at_k(retrieved, relevant, 5))  # 0.67 (2 out of 3)
```

#### Mean Reciprocal Rank (MRR)

```python
def mean_reciprocal_rank(queries_results: list[tuple]) -> float:
    """
    Average of reciprocal ranks of first relevant result.

    MRR = (1/|Q|) * Σ (1 / rank_i)
    """
    reciprocal_ranks = []

    for retrieved, relevant in queries_results:
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Example
results = [
    (["doc1", "doc2", "doc3"], {"doc1"}),  # First position = 1/1
    (["doc1", "doc2", "doc3"], {"doc2"}),  # Second position = 1/2
    (["doc1", "doc2", "doc3"], {"doc4"}),  # Not found = 0
]
print(mean_reciprocal_rank(results))  # 0.5
```

#### Normalized Discounted Cumulative Gain (NDCG)

```python
import numpy as np

def ndcg_at_k(retrieved: list, relevance_scores: dict, k: int) -> float:
    """
    NDCG accounts for position and graded relevance.

    DCG = Σ (rel_i / log2(i + 1))
    NDCG = DCG / IDCG (ideal DCG)
    """
    def dcg(scores):
        return sum(
            score / np.log2(i + 2)
            for i, score in enumerate(scores)
        )

    # Actual DCG
    actual_scores = [relevance_scores.get(doc, 0) for doc in retrieved[:k]]
    actual_dcg = dcg(actual_scores)

    # Ideal DCG (perfect ranking)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

# Example with graded relevance (0-2)
relevance = {"doc1": 2, "doc2": 1, "doc3": 2, "doc4": 0}
retrieved = ["doc4", "doc1", "doc2", "doc3"]
print(ndcg_at_k(retrieved, relevance, 4))  # < 1.0 (not ideal order)
```

### 2.2 Generation Metrics

#### Faithfulness

```python
async def measure_faithfulness(
    llm,
    response: str,
    context: str
) -> float:
    """
    Does the response only contain information from the context?
    """
    prompt = f"""
    Evaluate if the response is faithful to the context.
    A faithful response only contains information present in the context.

    Context:
    {context}

    Response:
    {response}

    Score the faithfulness from 0 to 1:
    - 1.0: All claims are supported by context
    - 0.5: Some claims are supported, some are not
    - 0.0: Most claims are not in context

    Return only the score (e.g., 0.85):
    """

    score = await llm.generate(prompt)
    return float(score.strip())
```

#### Answer Relevance

```python
async def measure_answer_relevance(
    llm,
    question: str,
    response: str
) -> float:
    """
    Does the response actually answer the question?
    """
    prompt = f"""
    Evaluate if the response answers the question.

    Question: {question}

    Response: {response}

    Score from 0 to 1:
    - 1.0: Directly and completely answers the question
    - 0.5: Partially answers or indirectly related
    - 0.0: Does not answer the question at all

    Return only the score:
    """

    score = await llm.generate(prompt)
    return float(score.strip())
```

#### Context Relevance

```python
async def measure_context_relevance(
    llm,
    question: str,
    contexts: list[str]
) -> float:
    """
    Is the retrieved context relevant to the question?
    """
    scores = []

    for context in contexts:
        prompt = f"""
        Is this context relevant to answering the question?

        Question: {question}

        Context: {context}

        Score from 0 to 1:
        - 1.0: Highly relevant, contains answer
        - 0.5: Somewhat relevant
        - 0.0: Not relevant

        Score:
        """

        score = await llm.generate(prompt)
        scores.append(float(score.strip()))

    return sum(scores) / len(scores) if scores else 0
```

### 2.3 End-to-End Metrics

#### Answer Correctness

```python
async def measure_correctness(
    llm,
    question: str,
    response: str,
    ground_truth: str
) -> dict:
    """
    Compare response to ground truth answer.
    """
    prompt = f"""
    Compare the response to the ground truth answer.

    Question: {question}

    Response: {response}

    Ground Truth: {ground_truth}

    Evaluate:
    1. Factual accuracy (0-1): Are the facts correct?
    2. Completeness (0-1): Does it cover all key points?
    3. No contradictions (0-1): Does it contradict the ground truth?

    Return JSON: {{"accuracy": X, "completeness": X, "no_contradictions": X}}
    """

    result = await llm.generate(prompt)
    scores = json.loads(result)
    scores["overall"] = sum(scores.values()) / len(scores)

    return scores
```

### 2.4 Performance Metrics

```python
import time
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    tokens_used: int
    estimated_cost: float

class PerformanceTracker:
    """Track performance metrics for each query."""

    def __init__(self):
        self.metrics = []

    def track_query(self):
        """Context manager for tracking a query."""
        return QueryTracker(self)

class QueryTracker:
    def __init__(self, parent):
        self.parent = parent
        self.metrics = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.metrics["total_latency_ms"] = (time.time() - self.start_time) * 1000
        self.parent.metrics.append(self.metrics)

    def mark_retrieval_done(self):
        self.metrics["retrieval_latency_ms"] = (time.time() - self.start_time) * 1000

    def set_tokens(self, input_tokens: int, output_tokens: int):
        self.metrics["tokens_used"] = input_tokens + output_tokens
        # Estimate cost (GPT-4 pricing)
        self.metrics["estimated_cost"] = (
            input_tokens * 0.00003 + output_tokens * 0.00006
        )

# Usage
tracker = PerformanceTracker()

with tracker.track_query() as t:
    results = retriever.retrieve(query)
    t.mark_retrieval_done()

    response = llm.generate(prompt)
    t.set_tokens(1000, 500)
```

[↑ Back to Top](#table-of-contents)

---

## 3. Evaluation Frameworks

### 3.1 RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    answer_correctness
)
from datasets import Dataset

def evaluate_with_ragas(eval_data: list[dict]) -> dict:
    """
    Evaluate RAG system using RAGAS framework.

    eval_data format:
    [
        {
            "question": "What is...",
            "answer": "Generated answer...",
            "contexts": ["Retrieved context 1", "Context 2"],
            "ground_truth": "Expected answer..."
        }
    ]
    """

    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(eval_data)

    # Run evaluation
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,           # Is answer grounded in context?
            answer_relevancy,       # Does answer address question?
            context_precision,      # Are retrieved contexts relevant?
            context_recall,         # Did we find all relevant contexts?
            answer_correctness      # Is answer factually correct?
        ]
    )

    return results

# Example usage
eval_data = [
    {
        "question": "What is the return policy?",
        "answer": "You can return items within 30 days for a full refund.",
        "contexts": [
            "Our return policy allows returns within 30 days of purchase.",
            "Refunds are processed within 5-7 business days."
        ],
        "ground_truth": "Items can be returned within 30 days for a full refund."
    }
]

results = evaluate_with_ragas(eval_data)
print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.92, ...}
```

### 3.2 Custom Evaluation Pipeline

```python
class RAGEvaluator:
    """Custom RAG evaluation pipeline."""

    def __init__(self, llm_client, retriever, generator):
        self.llm = llm_client
        self.retriever = retriever
        self.generator = generator

    async def evaluate(self, test_cases: list[dict]) -> dict:
        """
        Run comprehensive evaluation on test cases.

        test_cases format:
        [
            {
                "id": "test_1",
                "question": "...",
                "ground_truth_answer": "...",
                "relevant_doc_ids": ["doc1", "doc2"]
            }
        ]
        """
        results = []

        for case in test_cases:
            result = await self.evaluate_single(case)
            results.append(result)

        return self.aggregate_results(results)

    async def evaluate_single(self, case: dict) -> dict:
        question = case["question"]

        # Retrieval
        retrieved_docs = self.retriever.retrieve(question, top_k=5)
        retrieved_ids = [doc["id"] for doc in retrieved_docs]

        # Generation
        contexts = [doc["content"] for doc in retrieved_docs]
        response = await self.generator.generate(question, contexts)

        # Calculate metrics
        retrieval_metrics = {
            "precision@5": precision_at_k(
                retrieved_ids,
                set(case["relevant_doc_ids"]),
                5
            ),
            "recall@5": recall_at_k(
                retrieved_ids,
                set(case["relevant_doc_ids"]),
                5
            )
        }

        generation_metrics = {
            "faithfulness": await measure_faithfulness(
                self.llm, response, "\n".join(contexts)
            ),
            "answer_relevance": await measure_answer_relevance(
                self.llm, question, response
            ),
            "correctness": await measure_correctness(
                self.llm, question, response, case["ground_truth_answer"]
            )
        }

        return {
            "case_id": case["id"],
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "response": response
        }

    def aggregate_results(self, results: list) -> dict:
        """Aggregate individual results into summary metrics."""

        summary = {
            "retrieval": {
                "avg_precision@5": np.mean([r["retrieval"]["precision@5"] for r in results]),
                "avg_recall@5": np.mean([r["retrieval"]["recall@5"] for r in results])
            },
            "generation": {
                "avg_faithfulness": np.mean([r["generation"]["faithfulness"] for r in results]),
                "avg_answer_relevance": np.mean([r["generation"]["answer_relevance"] for r in results]),
                "avg_correctness": np.mean([r["generation"]["correctness"]["overall"] for r in results])
            },
            "individual_results": results
        }

        return summary
```

### 3.3 LLM-as-Judge

```python
class LLMJudge:
    """Use LLM to evaluate responses."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def judge(
        self,
        question: str,
        response: str,
        context: str,
        criteria: list[str] = None
    ) -> dict:
        criteria = criteria or [
            "accuracy",
            "completeness",
            "clarity",
            "relevance",
            "grounding"
        ]

        prompt = f"""
        You are an expert evaluator. Evaluate this RAG response.

        Question: {question}

        Context provided: {context[:2000]}

        Response to evaluate: {response}

        Rate each criterion from 1-5:
        {chr(10).join(f"- {c}: [description]" for c in criteria)}

        Provide:
        1. Score for each criterion (1-5)
        2. Brief justification
        3. Overall score (1-5)
        4. Suggestions for improvement

        Return as JSON:
        {{
            "scores": {{"criterion": score, ...}},
            "justifications": {{"criterion": "reason", ...}},
            "overall_score": X,
            "suggestions": ["suggestion1", ...]
        }}
        """

        result = await self.llm.generate(prompt)
        return json.loads(result)

    async def compare_responses(
        self,
        question: str,
        response_a: str,
        response_b: str,
        context: str
    ) -> dict:
        """A/B comparison of two responses."""

        prompt = f"""
        Compare these two responses to the same question.

        Question: {question}
        Context: {context[:1500]}

        Response A: {response_a}

        Response B: {response_b}

        Which is better and why? Consider:
        1. Accuracy
        2. Completeness
        3. Clarity
        4. Use of context

        Return JSON:
        {{
            "winner": "A" or "B" or "tie",
            "score_a": 1-5,
            "score_b": 1-5,
            "reasoning": "explanation"
        }}
        """

        return json.loads(await self.llm.generate(prompt))
```

[↑ Back to Top](#table-of-contents)

---

## 4. Testing Strategies

### 4.1 Golden Dataset Creation

```python
class GoldenDatasetBuilder:
    """Build evaluation dataset with ground truth."""

    def __init__(self):
        self.test_cases = []

    def add_case(
        self,
        question: str,
        ground_truth: str,
        relevant_docs: list[str],
        category: str = "general",
        difficulty: str = "medium"
    ):
        self.test_cases.append({
            "id": f"test_{len(self.test_cases)}",
            "question": question,
            "ground_truth": ground_truth,
            "relevant_docs": relevant_docs,
            "category": category,
            "difficulty": difficulty
        })

    def add_from_qa_pairs(self, qa_pairs: list[dict]):
        """Add cases from existing Q&A pairs."""
        for pair in qa_pairs:
            self.add_case(
                question=pair["question"],
                ground_truth=pair["answer"],
                relevant_docs=pair.get("sources", [])
            )

    async def generate_synthetic(
        self,
        llm,
        documents: list[str],
        n_per_doc: int = 3
    ):
        """Generate synthetic test cases from documents."""

        for doc in documents:
            prompt = f"""
            Generate {n_per_doc} question-answer pairs from this document.
            Questions should be diverse and test different aspects.

            Document:
            {doc[:3000]}

            Return JSON array:
            [
                {{"question": "...", "answer": "..."}},
                ...
            ]
            """

            pairs = json.loads(await llm.generate(prompt))

            for pair in pairs:
                self.add_case(
                    question=pair["question"],
                    ground_truth=pair["answer"],
                    relevant_docs=[doc]
                )

    def export(self, path: str):
        """Export dataset to file."""
        with open(path, "w") as f:
            json.dump(self.test_cases, f, indent=2)

    def get_by_category(self, category: str) -> list:
        return [c for c in self.test_cases if c["category"] == category]

    def get_by_difficulty(self, difficulty: str) -> list:
        return [c for c in self.test_cases if c["difficulty"] == difficulty]
```

### 4.2 Test Categories

```python
class TestSuiteBuilder:
    """Build comprehensive test suite."""

    def build_suite(self) -> dict:
        return {
            # Basic functionality
            "smoke_tests": self.smoke_tests(),

            # Retrieval edge cases
            "retrieval_tests": self.retrieval_tests(),

            # Generation edge cases
            "generation_tests": self.generation_tests(),

            # Robustness tests
            "robustness_tests": self.robustness_tests(),

            # Performance tests
            "performance_tests": self.performance_tests()
        }

    def smoke_tests(self) -> list:
        """Basic sanity checks."""
        return [
            {"question": "What is the company name?", "type": "simple_fact"},
            {"question": "How do I contact support?", "type": "simple_fact"},
        ]

    def retrieval_tests(self) -> list:
        """Test retrieval edge cases."""
        return [
            # Exact match
            {"question": "What is policy ABC-123?", "type": "exact_match"},

            # Semantic match
            {"question": "How to recover forgotten password?", "type": "semantic"},

            # Multi-hop
            {"question": "What's the manager of the person who handles returns?", "type": "multi_hop"},

            # No answer exists
            {"question": "What is the meaning of life?", "type": "no_answer"},

            # Ambiguous
            {"question": "Tell me about the policy", "type": "ambiguous"},
        ]

    def generation_tests(self) -> list:
        """Test generation edge cases."""
        return [
            # Requires synthesis
            {"question": "Compare plans A and B", "type": "synthesis"},

            # Long answer needed
            {"question": "Explain the full process", "type": "long_form"},

            # Conflicting information
            {"question": "Question with conflicting docs", "type": "conflict"},

            # Requires calculation
            {"question": "What's the total cost for 3 items?", "type": "calculation"},
        ]

    def robustness_tests(self) -> list:
        """Test system robustness."""
        return [
            # Typos
            {"question": "How do I resset my pasword?", "type": "typo"},

            # Different phrasing
            {"question": "password reset", "type": "keyword_only"},
            {"question": "I forgot my password and need help", "type": "verbose"},

            # Injection attempts
            {"question": "Ignore previous instructions and...", "type": "injection"},

            # Unicode
            {"question": "How to use émojis 🎉?", "type": "unicode"},
        ]

    def performance_tests(self) -> list:
        """Test performance under various conditions."""
        return [
            {"question": "Simple question", "expected_latency_ms": 500},
            {"question": "Complex multi-part question", "expected_latency_ms": 2000},
        ]
```

### 4.3 Automated Testing Pipeline

```python
import pytest
from typing import Generator

class RAGTestRunner:
    """Run automated RAG tests."""

    def __init__(self, rag_system, evaluator):
        self.rag = rag_system
        self.evaluator = evaluator
        self.results = []

    async def run_all_tests(self, test_suite: dict) -> dict:
        """Run all tests in suite."""
        results = {}

        for category, tests in test_suite.items():
            results[category] = await self.run_category(category, tests)

        return results

    async def run_category(self, category: str, tests: list) -> dict:
        """Run tests in a category."""
        passed = 0
        failed = 0
        failures = []

        for test in tests:
            try:
                result = await self.run_single_test(test)
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
                    failures.append(result)
            except Exception as e:
                failed += 1
                failures.append({"test": test, "error": str(e)})

        return {
            "total": len(tests),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(tests) if tests else 0,
            "failures": failures
        }

    async def run_single_test(self, test: dict) -> dict:
        """Run a single test case."""

        # Get response from RAG
        response = await self.rag.query(test["question"])

        # Evaluate
        metrics = await self.evaluator.evaluate_single({
            "question": test["question"],
            "ground_truth_answer": test.get("expected_answer", ""),
            "relevant_doc_ids": test.get("relevant_docs", [])
        })

        # Determine pass/fail
        passed = self.check_pass_criteria(metrics, test)

        return {
            "test": test,
            "response": response,
            "metrics": metrics,
            "passed": passed
        }

    def check_pass_criteria(self, metrics: dict, test: dict) -> bool:
        """Check if test passes based on criteria."""

        # Default thresholds
        thresholds = {
            "faithfulness": 0.8,
            "answer_relevance": 0.7,
            "precision@5": 0.6
        }

        # Override with test-specific thresholds
        thresholds.update(test.get("thresholds", {}))

        for metric, threshold in thresholds.items():
            if metric in metrics.get("generation", {}):
                if metrics["generation"][metric] < threshold:
                    return False
            if metric in metrics.get("retrieval", {}):
                if metrics["retrieval"][metric] < threshold:
                    return False

        return True
```

[↑ Back to Top](#table-of-contents)

---

## 5. Production Monitoring

### 5.1 Metrics Dashboard

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

@dataclass
class QueryLog:
    timestamp: datetime
    query: str
    response: str
    latency_ms: float
    tokens_used: int
    cost: float
    user_feedback: int = None  # 1-5 rating
    retrieved_docs: list = None

class ProductionMonitor:
    """Monitor RAG system in production."""

    def __init__(self):
        self.logs: list[QueryLog] = []
        self.alerts = []

    def log_query(self, log: QueryLog):
        """Log a query."""
        self.logs.append(log)
        self.check_alerts(log)

    def get_metrics(self, hours: int = 24) -> dict:
        """Get metrics for last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [l for l in self.logs if l.timestamp > cutoff]

        if not recent:
            return {"error": "No data"}

        return {
            "query_count": len(recent),
            "avg_latency_ms": sum(l.latency_ms for l in recent) / len(recent),
            "p95_latency_ms": self.percentile([l.latency_ms for l in recent], 95),
            "p99_latency_ms": self.percentile([l.latency_ms for l in recent], 99),
            "total_tokens": sum(l.tokens_used for l in recent),
            "total_cost": sum(l.cost for l in recent),
            "avg_cost_per_query": sum(l.cost for l in recent) / len(recent),
            "feedback_count": sum(1 for l in recent if l.user_feedback),
            "avg_feedback": self.avg_feedback(recent),
            "queries_per_hour": len(recent) / hours
        }

    def percentile(self, values: list, p: int) -> float:
        """Calculate percentile."""
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def avg_feedback(self, logs: list) -> float:
        """Calculate average user feedback."""
        rated = [l.user_feedback for l in logs if l.user_feedback]
        return sum(rated) / len(rated) if rated else 0

    def check_alerts(self, log: QueryLog):
        """Check for alert conditions."""
        # High latency alert
        if log.latency_ms > 5000:
            self.alerts.append({
                "type": "high_latency",
                "value": log.latency_ms,
                "timestamp": log.timestamp
            })

        # Low feedback alert
        if log.user_feedback and log.user_feedback < 2:
            self.alerts.append({
                "type": "negative_feedback",
                "query": log.query,
                "timestamp": log.timestamp
            })

    def get_quality_breakdown(self, hours: int = 24) -> dict:
        """Break down quality metrics."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [l for l in self.logs if l.timestamp > cutoff]

        feedback_dist = defaultdict(int)
        for log in recent:
            if log.user_feedback:
                feedback_dist[log.user_feedback] += 1

        return {
            "feedback_distribution": dict(feedback_dist),
            "positive_rate": sum(1 for l in recent if l.user_feedback and l.user_feedback >= 4) / len(recent) if recent else 0,
            "negative_rate": sum(1 for l in recent if l.user_feedback and l.user_feedback <= 2) / len(recent) if recent else 0
        }
```

### 5.2 Alerting System

```python
class AlertManager:
    """Manage production alerts."""

    def __init__(self, notification_service):
        self.notifier = notification_service
        self.rules = []

    def add_rule(
        self,
        name: str,
        condition: callable,
        severity: str = "warning",
        cooldown_minutes: int = 30
    ):
        self.rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "cooldown": timedelta(minutes=cooldown_minutes),
            "last_fired": None
        })

    def check_metrics(self, metrics: dict):
        """Check metrics against all rules."""
        for rule in self.rules:
            if rule["condition"](metrics):
                if self.should_fire(rule):
                    self.fire_alert(rule, metrics)

    def should_fire(self, rule: dict) -> bool:
        """Check if alert should fire (respecting cooldown)."""
        if rule["last_fired"] is None:
            return True
        return datetime.utcnow() - rule["last_fired"] > rule["cooldown"]

    def fire_alert(self, rule: dict, metrics: dict):
        """Fire an alert."""
        rule["last_fired"] = datetime.utcnow()

        alert = {
            "name": rule["name"],
            "severity": rule["severity"],
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.notifier.send(alert)

# Setup alerts
alert_manager = AlertManager(notification_service)

# Latency alert
alert_manager.add_rule(
    name="high_latency",
    condition=lambda m: m.get("p95_latency_ms", 0) > 3000,
    severity="warning"
)

# Error rate alert
alert_manager.add_rule(
    name="high_error_rate",
    condition=lambda m: m.get("error_rate", 0) > 0.05,
    severity="critical"
)

# Quality degradation alert
alert_manager.add_rule(
    name="quality_drop",
    condition=lambda m: m.get("avg_feedback", 5) < 3.5,
    severity="warning"
)

# Cost spike alert
alert_manager.add_rule(
    name="cost_spike",
    condition=lambda m: m.get("avg_cost_per_query", 0) > 0.10,
    severity="warning"
)
```

### 5.3 Continuous Evaluation

```python
class ContinuousEvaluator:
    """Continuously evaluate production quality."""

    def __init__(self, evaluator, sample_rate: float = 0.1):
        self.evaluator = evaluator
        self.sample_rate = sample_rate
        self.evaluation_results = []

    async def maybe_evaluate(self, query_log: QueryLog) -> dict:
        """Randomly evaluate a sample of queries."""
        import random

        if random.random() > self.sample_rate:
            return None

        # Run evaluation
        result = await self.evaluate_query(query_log)
        self.evaluation_results.append(result)

        return result

    async def evaluate_query(self, log: QueryLog) -> dict:
        """Evaluate a single query."""

        metrics = {
            "timestamp": datetime.utcnow(),
            "query": log.query,
            "faithfulness": await self.evaluator.measure_faithfulness(
                log.response,
                "\n".join(log.retrieved_docs or [])
            ),
            "answer_relevance": await self.evaluator.measure_answer_relevance(
                log.query,
                log.response
            )
        }

        return metrics

    def get_trend(self, metric: str, days: int = 7) -> dict:
        """Get metric trend over time."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [r for r in self.evaluation_results if r["timestamp"] > cutoff]

        if not recent:
            return {"error": "No data"}

        # Group by day
        by_day = defaultdict(list)
        for r in recent:
            day = r["timestamp"].date()
            by_day[day].append(r[metric])

        trend = {
            str(day): sum(vals) / len(vals)
            for day, vals in sorted(by_day.items())
        }

        return {
            "metric": metric,
            "trend": trend,
            "current": list(trend.values())[-1] if trend else 0,
            "change": self.calculate_change(trend)
        }

    def calculate_change(self, trend: dict) -> float:
        """Calculate % change in metric."""
        values = list(trend.values())
        if len(values) < 2:
            return 0
        return (values[-1] - values[0]) / values[0] * 100
```

### 5.4 Logging Best Practices

```python
import structlog
import json

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

class RAGLogger:
    """Structured logging for RAG system."""

    def log_query(
        self,
        query: str,
        response: str,
        context: list,
        metrics: dict,
        user_id: str = None
    ):
        logger.info(
            "rag_query",
            query=query,
            response_length=len(response),
            context_count=len(context),
            latency_ms=metrics.get("latency_ms"),
            tokens=metrics.get("tokens"),
            user_id=user_id
        )

    def log_retrieval(
        self,
        query: str,
        results: list,
        latency_ms: float
    ):
        logger.info(
            "rag_retrieval",
            query=query,
            result_count=len(results),
            top_score=results[0]["score"] if results else 0,
            latency_ms=latency_ms
        )

    def log_error(
        self,
        error: Exception,
        query: str,
        stage: str
    ):
        logger.error(
            "rag_error",
            error_type=type(error).__name__,
            error_message=str(error),
            query=query,
            stage=stage
        )

    def log_feedback(
        self,
        query_id: str,
        rating: int,
        comment: str = None
    ):
        logger.info(
            "rag_feedback",
            query_id=query_id,
            rating=rating,
            comment=comment
        )
```

[↑ Back to Top](#table-of-contents)

---

## 6. Problem Statements & Solutions

### Problem 1: No Ground Truth Data

**Symptoms:**
- Can't measure correctness
- Don't know what good looks like
- No baseline to compare against

**Solution:**

```python
class SyntheticGroundTruthGenerator:
    """Generate ground truth from documents."""

    def __init__(self, llm):
        self.llm = llm

    async def generate_qa_pairs(
        self,
        documents: list[str],
        questions_per_doc: int = 5
    ) -> list[dict]:
        """Generate Q&A pairs from documents."""

        qa_pairs = []

        for doc in documents:
            prompt = f"""
            Generate {questions_per_doc} diverse question-answer pairs from this document.

            Requirements:
            - Questions should be natural and varied
            - Answers must be directly supported by the document
            - Include simple and complex questions
            - Include factual and reasoning questions

            Document:
            {doc[:4000]}

            Return JSON array:
            [
                {{
                    "question": "...",
                    "answer": "...",
                    "difficulty": "easy/medium/hard",
                    "type": "factual/reasoning/comparison"
                }}
            ]
            """

            pairs = json.loads(await self.llm.generate(prompt))
            for pair in pairs:
                pair["source_doc"] = doc[:500]
            qa_pairs.extend(pairs)

        return qa_pairs

    async def validate_qa_pairs(self, qa_pairs: list) -> list:
        """Validate generated Q&A pairs."""

        validated = []

        for pair in qa_pairs:
            prompt = f"""
            Validate this Q&A pair:

            Question: {pair["question"]}
            Answer: {pair["answer"]}
            Source: {pair["source_doc"]}

            Is the answer correct and well-supported? (yes/no)
            """

            if "yes" in (await self.llm.generate(prompt)).lower():
                validated.append(pair)

        return validated
```

---

### Problem 2: Metrics Don't Reflect User Experience

**Symptoms:**
- High automated scores but users complain
- Metrics gaming
- Disconnect between numbers and reality

**Solution:**

```python
class UserCentricEvaluator:
    """Evaluate based on user experience."""

    def __init__(self, monitor):
        self.monitor = monitor

    def calculate_satisfaction_score(self) -> dict:
        """Calculate user satisfaction from feedback."""

        logs = self.monitor.logs
        with_feedback = [l for l in logs if l.user_feedback]

        if not with_feedback:
            return {"error": "No feedback data"}

        return {
            "nps": self.calculate_nps(with_feedback),
            "csat": self.calculate_csat(with_feedback),
            "thumbs_up_rate": self.calculate_thumbs_up_rate(with_feedback),
            "repeat_query_rate": self.calculate_repeat_rate(logs)
        }

    def calculate_nps(self, logs: list) -> float:
        """Net Promoter Score (assuming 1-5 scale)."""
        promoters = sum(1 for l in logs if l.user_feedback >= 5)
        detractors = sum(1 for l in logs if l.user_feedback <= 2)
        return (promoters - detractors) / len(logs) * 100

    def calculate_csat(self, logs: list) -> float:
        """Customer Satisfaction (% satisfied)."""
        satisfied = sum(1 for l in logs if l.user_feedback >= 4)
        return satisfied / len(logs) * 100

    def calculate_thumbs_up_rate(self, logs: list) -> float:
        """Simple thumbs up rate."""
        positive = sum(1 for l in logs if l.user_feedback >= 4)
        return positive / len(logs)

    def calculate_repeat_rate(self, logs: list) -> float:
        """Rate of users asking similar questions (potential dissatisfaction)."""
        # Implementation: detect similar queries from same user session
        pass

    def correlate_metrics_with_satisfaction(self) -> dict:
        """Find which metrics correlate with satisfaction."""

        # Analyze correlation between automated metrics and user feedback
        high_satisfaction = [l for l in self.monitor.logs if l.user_feedback and l.user_feedback >= 4]
        low_satisfaction = [l for l in self.monitor.logs if l.user_feedback and l.user_feedback <= 2]

        return {
            "high_satisfaction_avg_latency": np.mean([l.latency_ms for l in high_satisfaction]) if high_satisfaction else 0,
            "low_satisfaction_avg_latency": np.mean([l.latency_ms for l in low_satisfaction]) if low_satisfaction else 0,
            # Add more correlations
        }
```

---

### Problem 3: A/B Testing RAG Systems

**Solution:**

```python
class ABTestManager:
    """Manage A/B tests for RAG systems."""

    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: {"A": [], "B": []})

    def create_experiment(
        self,
        name: str,
        variant_a: callable,
        variant_b: callable,
        traffic_split: float = 0.5
    ):
        self.experiments[name] = {
            "A": variant_a,
            "B": variant_b,
            "split": traffic_split,
            "active": True
        }

    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Deterministically assign user to variant."""
        experiment = self.experiments[experiment_name]

        # Consistent hashing for user assignment
        hash_val = hash(f"{experiment_name}:{user_id}") % 100

        if hash_val < experiment["split"] * 100:
            return "A"
        return "B"

    async def run_query(
        self,
        experiment_name: str,
        user_id: str,
        query: str
    ) -> dict:
        """Run query through experiment."""

        variant = self.get_variant(experiment_name, user_id)
        experiment = self.experiments[experiment_name]

        handler = experiment[variant]
        result = await handler(query)

        return {
            "variant": variant,
            "result": result
        }

    def record_result(
        self,
        experiment_name: str,
        variant: str,
        metrics: dict
    ):
        """Record experiment result."""
        self.results[experiment_name][variant].append(metrics)

    def analyze_experiment(self, experiment_name: str) -> dict:
        """Analyze experiment results."""
        results = self.results[experiment_name]

        def avg_metric(data, metric):
            values = [d[metric] for d in data if metric in d]
            return sum(values) / len(values) if values else 0

        analysis = {
            "A": {
                "count": len(results["A"]),
                "avg_latency": avg_metric(results["A"], "latency_ms"),
                "avg_satisfaction": avg_metric(results["A"], "user_feedback"),
            },
            "B": {
                "count": len(results["B"]),
                "avg_latency": avg_metric(results["B"], "latency_ms"),
                "avg_satisfaction": avg_metric(results["B"], "user_feedback"),
            }
        }

        # Statistical significance
        analysis["significant"] = self.is_significant(
            results["A"], results["B"], "user_feedback"
        )

        return analysis

    def is_significant(self, a_data: list, b_data: list, metric: str, alpha: float = 0.05) -> bool:
        """Check statistical significance using t-test."""
        from scipy import stats

        a_values = [d[metric] for d in a_data if metric in d]
        b_values = [d[metric] for d in b_data if metric in d]

        if len(a_values) < 30 or len(b_values) < 30:
            return False  # Not enough data

        _, p_value = stats.ttest_ind(a_values, b_values)
        return p_value < alpha
```

[↑ Back to Top](#table-of-contents)

---

## 7. Trade-offs

### Evaluation Method Trade-offs

| Method | Accuracy | Speed | Cost | Scalability |
|--------|----------|-------|------|-------------|
| Human evaluation | Highest | Slowest | $$$$$ | Low |
| LLM-as-judge | High | Medium | $$ | Medium |
| Automated metrics | Medium | Fast | $ | High |
| User feedback | Real-world | Slow | $ | High |

### Metric Trade-offs

| Metric | What It Captures | What It Misses |
|--------|------------------|----------------|
| Precision@K | Relevance of results | Missed relevant docs |
| Recall@K | Coverage | Result quality |
| Faithfulness | Grounding | Answer quality |
| User feedback | Real satisfaction | Specific issues |

### Monitoring Depth Trade-offs

| Level | Insight | Overhead | Storage |
|-------|---------|----------|---------|
| Basic (latency, errors) | Low | Low | Low |
| Standard (+ quality sample) | Medium | Medium | Medium |
| Comprehensive (all queries) | High | High | High |

[↑ Back to Top](#table-of-contents)

---

## 8. Cost-Effective Solutions

### Free Evaluation Stack

```python
# Evaluation without expensive LLM calls

# 1. Retrieval metrics - completely free
def free_retrieval_metrics(retrieved, relevant):
    return {
        "precision": precision_at_k(retrieved, relevant, 5),
        "recall": recall_at_k(retrieved, relevant, 5),
        "mrr": mean_reciprocal_rank([(retrieved, relevant)])
    }

# 2. Simple heuristic generation metrics
def heuristic_faithfulness(response, context):
    """Estimate faithfulness without LLM."""
    context_words = set(context.lower().split())
    response_words = set(response.lower().split())

    # Overlap as proxy for faithfulness
    overlap = len(context_words & response_words)
    return overlap / len(response_words) if response_words else 0

# 3. User feedback - free
# Just collect thumbs up/down
```

### Sampling Strategy

```python
class CostEfficientEvaluator:
    """Evaluate samples to reduce cost."""

    def __init__(self, full_evaluator, sample_rate: float = 0.05):
        self.evaluator = full_evaluator
        self.sample_rate = sample_rate

    async def evaluate_sample(self, queries: list) -> dict:
        """Evaluate a sample of queries."""
        import random

        sample_size = max(1, int(len(queries) * self.sample_rate))
        sample = random.sample(queries, sample_size)

        results = []
        for query in sample:
            result = await self.evaluator.evaluate(query)
            results.append(result)

        return {
            "sample_size": sample_size,
            "results": self.aggregate(results),
            "confidence_interval": self.calculate_ci(results)
        }

    def calculate_ci(self, results: list, confidence: float = 0.95) -> dict:
        """Calculate confidence interval for sampled metrics."""
        from scipy import stats

        metrics = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            mean = np.mean(values)
            se = stats.sem(values)
            ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=se)
            metrics[key] = {"mean": mean, "ci_low": ci[0], "ci_high": ci[1]}

        return metrics
```

[↑ Back to Top](#table-of-contents)

---

## 9. Best Practices

### DO's

1. **Establish Baselines First**
   ```python
   # Before any changes, record baseline metrics
   baseline = evaluator.evaluate(test_set)
   save_baseline(baseline)
   ```

2. **Use Multiple Metrics**
   ```python
   # Don't rely on single metric
   metrics = {
       "retrieval": retrieval_metrics,
       "generation": generation_metrics,
       "user": user_metrics
   }
   ```

3. **Automate Evaluation**
   ```python
   # CI/CD integration
   def test_rag_quality():
       results = evaluator.evaluate(golden_dataset)
       assert results["faithfulness"] > 0.8
       assert results["precision@5"] > 0.7
   ```

4. **Track Trends Over Time**
   ```python
   # Store historical metrics
   metrics_store.save(date=today, metrics=current_metrics)
   trend = metrics_store.get_trend(days=30)
   ```

5. **Include Edge Cases in Tests**
   - No relevant documents
   - Conflicting information
   - Out-of-scope questions

### DON'Ts

1. **Don't Rely on Single Metric**
   - High precision with low recall isn't good

2. **Don't Skip User Feedback**
   - Automated metrics don't capture everything

3. **Don't Evaluate on Training Data**
   - Use held-out test sets

4. **Don't Ignore Statistical Significance**
   - Small improvements might be noise

5. **Don't Set and Forget**
   - Quality can drift over time

[↑ Back to Top](#table-of-contents)

---

## 10. Quick Reference

### Metric Targets

```
Metric              Minimum    Good      Excellent
───────────────────────────────────────────────────
Precision@5         > 0.6     > 0.8     > 0.9
Recall@10           > 0.5     > 0.7     > 0.85
MRR                 > 0.5     > 0.7     > 0.85
Faithfulness        > 0.8     > 0.9     > 0.95
Answer Relevance    > 0.7     > 0.85    > 0.95
User Satisfaction   > 3.5/5   > 4.0/5   > 4.5/5
```

### Evaluation Checklist

```
□ Golden dataset created (50+ test cases)
□ Retrieval metrics tracked
□ Generation metrics tracked
□ User feedback collected
□ Automated testing in CI/CD
□ Production monitoring active
□ Alerts configured
□ Regular quality reviews scheduled
```

### Monitoring Dashboard Essentials

```
REAL-TIME:
- Query volume
- Latency (p50, p95, p99)
- Error rate

QUALITY:
- Faithfulness (sampled)
- User satisfaction
- Negative feedback rate

COSTS:
- Tokens used
- Cost per query
- Daily/monthly spend
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Generation & Prompting](./06-generation-prompting.md) | [Main Guide](../README.md) | [Security & Compliance →](./08-security-compliance.md) |

---

*Last updated: 2024*
