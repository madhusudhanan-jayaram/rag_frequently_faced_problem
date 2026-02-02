# Chunking Strategies for RAG

Complete guide to splitting documents into optimal chunks for retrieval.

[← Back to Main Guide](../README.md) | [← Previous: Data Cleanup](./01-data-cleanup.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Chunk Size Guidelines](#2-chunk-size-guidelines)
- [3. Chunking Methods](#3-chunking-methods)
- [4. Overlap Strategies](#4-overlap-strategies)
- [5. Advanced Chunking](#5-advanced-chunking)
- [6. Problem Statements & Solutions](#6-problem-statements--solutions)
- [7. Trade-offs](#7-trade-offs)
- [8. Cost-Effective Solutions](#8-cost-effective-solutions)
- [9. Best Practices](#9-best-practices)
- [10. Quick Reference](#10-quick-reference)

---

## 1. Overview

### Why Chunking Matters

Chunking is the process of splitting documents into smaller pieces for embedding and retrieval. It's one of the most impactful decisions in RAG system design.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Impact of Chunk Size                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Too Small                            Too Large                     │
│  ──────────                           ─────────                     │
│  • Lost context                       • Diluted relevance           │
│  • Fragmented info                    • Harder to match             │
│  • More chunks to search              • Wastes context window       │
│  • Higher precision                   • Higher recall               │
│  • Lower recall                       • Lower precision             │
│                                                                     │
│                        Just Right                                   │
│                        ──────────                                   │
│                   • Complete thoughts                               │
│                   • Semantic coherence                              │
│                   • Efficient retrieval                             │
│                   • Balanced precision/recall                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Term | Definition |
|------|------------|
| **Chunk** | A segment of text stored and retrieved as a unit |
| **Chunk Size** | Length of chunk (characters, tokens, or words) |
| **Overlap** | Shared content between adjacent chunks |
| **Stride** | Distance between chunk start positions |
| **Token** | Unit used by LLMs (roughly 4 characters in English) |

[↑ Back to Top](#table-of-contents)

---

## 2. Chunk Size Guidelines

### 2.1 Size Recommendations by Use Case

| Category | Token Range | Character Range | Use Cases |
|----------|-------------|-----------------|-----------|
| **Micro** | 64-128 | 250-500 | Definitions, FAQs, factoids |
| **Small** | 128-256 | 500-1000 | Q&A, support tickets, short answers |
| **Medium** | 256-512 | 1000-2000 | General documents, articles |
| **Large** | 512-1024 | 2000-4000 | Technical docs, procedures |
| **Extra Large** | 1024-2048 | 4000-8000 | Research papers, legal documents |

### 2.2 Recommendations by Document Type

| Document Type | Recommended Size | Overlap | Rationale |
|---------------|------------------|---------|-----------|
| **FAQ / Q&A** | 128-256 tokens | 20-30 | One Q&A per chunk |
| **Knowledge Base** | 256-512 tokens | 50-100 | Topic-focused sections |
| **Technical Docs** | 512-768 tokens | 100-150 | Complete procedures |
| **API Documentation** | 256-512 tokens | 50-100 | Function-level chunks |
| **Legal Contracts** | 512-1024 tokens | 150-200 | Clause integrity |
| **Research Papers** | 768-1024 tokens | 150-200 | Methodology/findings together |
| **Customer Support** | 128-256 tokens | 20-50 | Specific issue resolution |
| **Meeting Notes** | 512-768 tokens | 100-150 | Discussion continuity |
| **Product Manuals** | 512-768 tokens | 100-150 | Step-by-step integrity |
| **Email Threads** | 256-384 tokens | 50-75 | Message context |
| **Code Files** | 256-512 tokens | 50-100 | Function-level |
| **Chat Logs** | 256-512 tokens | 50-100 | Conversation context |

### 2.3 Chunk Size Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│               Chunk Size Decision Tree                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  What's your primary goal?                                       │
│  │                                                               │
│  ├─► Precise, factual answers                                    │
│  │   └─► Small chunks (128-256 tokens)                          │
│  │       Examples: FAQ, definitions, lookups                     │
│  │                                                               │
│  ├─► Balanced context and precision                              │
│  │   └─► Medium chunks (256-512 tokens)                         │
│  │       Examples: Knowledge base, general docs                  │
│  │                                                               │
│  ├─► Rich context for complex questions                          │
│  │   └─► Large chunks (512-1024 tokens)                         │
│  │       Examples: Technical docs, analysis                      │
│  │                                                               │
│  └─► Comprehensive understanding                                 │
│      └─► Extra large chunks (1024-2048 tokens)                  │
│          Examples: Research, legal                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

[↑ Back to Top](#table-of-contents)

---

## 3. Chunking Methods

### 3.1 Fixed-Size Chunking

The simplest approach: split text into chunks of fixed length.

```python
from langchain.text_splitter import CharacterTextSplitter

# Character-based fixed chunking
splitter = CharacterTextSplitter(
    chunk_size=1000,      # characters
    chunk_overlap=200,    # character overlap
    separator="\n",       # prefer splitting at newlines
    length_function=len,  # use character count
)

chunks = splitter.split_text(document)
```

**Token-based fixed chunking:**

```python
import tiktoken

class TokenBasedChunker:
    """Fixed-size chunking using token count."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size

            # Get chunk tokens
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position (accounting for overlap)
            start = end - self.overlap

        return chunks
```

| Pros | Cons |
|------|------|
| Simple implementation | Ignores semantic boundaries |
| Predictable chunk sizes | May split mid-sentence |
| Fast processing | May split mid-word (character-based) |
| Easy to tune | Context fragmentation |

**When to Use:**
- Prototyping
- Uniform content (logs, structured data)
- When speed matters more than quality

---

### 3.2 Semantic Chunking

Split based on semantic similarity between sentences.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Semantic chunking with similarity threshold
splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95,          # percentile threshold
)

chunks = splitter.split_text(document)
```

**Custom semantic chunker:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    """Split text based on semantic similarity."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        # Split into sentences
        sentences = self.split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Get embeddings
        embeddings = self.model.encode(sentences)

        # Find breakpoints based on similarity drops
        breakpoints = [0]
        for i in range(1, len(embeddings)):
            similarity = np.dot(embeddings[i-1], embeddings[i])

            if similarity < self.similarity_threshold:
                breakpoints.append(i)

        breakpoints.append(len(sentences))

        # Create chunks
        chunks = []
        for i in range(len(breakpoints) - 1):
            chunk_sentences = sentences[breakpoints[i]:breakpoints[i+1]]
            chunk_text = " ".join(chunk_sentences)

            # Handle size constraints
            if len(chunk_text) > self.max_chunk_size:
                # Split large chunks
                chunks.extend(self.split_large_chunk(chunk_text))
            elif len(chunk_text) < self.min_chunk_size and chunks:
                # Merge small chunks with previous
                chunks[-1] += " " + chunk_text
            else:
                chunks.append(chunk_text)

        return chunks

    def split_into_sentences(self, text: str) -> list[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_large_chunk(self, text: str) -> list[str]:
        # Fall back to fixed-size for oversized chunks
        words = text.split()
        chunks = []
        current = []

        for word in words:
            current.append(word)
            if len(" ".join(current)) >= self.max_chunk_size:
                chunks.append(" ".join(current))
                current = []

        if current:
            chunks.append(" ".join(current))

        return chunks
```

| Pros | Cons |
|------|------|
| Preserves meaning | Requires embedding calls |
| Better retrieval quality | Variable chunk sizes |
| Natural breakpoints | More complex |
| Context coherence | Slower processing |

**When to Use:**
- Quality-critical applications
- Narrative content
- Documents with topic shifts

---

### 3.3 Recursive Character Splitting

Hierarchically split using multiple separators.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Default separators: ["\n\n", "\n", " ", ""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

chunks = splitter.split_text(document)
```

**Custom recursive splitter:**

```python
class RecursiveChunker:
    """Recursively split text using separator hierarchy."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " ", ""]

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        # Base case: text is small enough
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try each separator
        for i, separator in enumerate(separators):
            if separator in text:
                splits = text.split(separator)

                chunks = []
                current_chunk = ""

                for split in splits:
                    # Add separator back (except for last split)
                    piece = split + separator if separator else split

                    if len(current_chunk) + len(piece) <= self.chunk_size:
                        current_chunk += piece
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())

                        # If piece is too large, recursively split
                        if len(piece) > self.chunk_size:
                            sub_chunks = self._split(piece, separators[i+1:])
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = piece

                if current_chunk:
                    chunks.append(current_chunk.strip())

                return chunks

        # No separator found, force split by characters
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
```

| Pros | Cons |
|------|------|
| Respects document hierarchy | Still character-based at core |
| Better than pure fixed-size | May still split sentences |
| Very popular, well-tested | Requires separator tuning |
| Good balance | Doesn't understand semantics |

**When to Use:**
- Most general use cases
- Structured documents
- When you need reliability

---

### 3.4 Document-Aware Chunking

Split based on document structure (headers, sections, paragraphs).

```python
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# Parse and chunk by sections
elements = partition(filename="document.pdf")

chunks = chunk_by_title(
    elements,
    max_characters=1500,
    new_after_n_chars=1000,
    combine_text_under_n_chars=200,
    multipage_sections=True
)
```

**Custom document-aware chunker:**

```python
import re
from dataclasses import dataclass

@dataclass
class Section:
    title: str
    level: int
    content: str
    subsections: list

class DocumentAwareChunker:
    """Chunk based on document structure."""

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        include_title_in_chunk: bool = True
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_title_in_chunk = include_title_in_chunk

        # Header patterns (Markdown-style)
        self.header_patterns = [
            (r'^#{1}\s+(.+)$', 1),   # # Header
            (r'^#{2}\s+(.+)$', 2),   # ## Header
            (r'^#{3}\s+(.+)$', 3),   # ### Header
            (r'^#{4}\s+(.+)$', 4),   # #### Header
        ]

    def chunk(self, text: str) -> list[dict]:
        # Parse document structure
        sections = self.parse_structure(text)

        # Convert sections to chunks
        chunks = []
        for section in sections:
            section_chunks = self.section_to_chunks(section)
            chunks.extend(section_chunks)

        return chunks

    def parse_structure(self, text: str) -> list[Section]:
        lines = text.split("\n")
        sections = []
        current_section = None
        current_content = []

        for line in lines:
            header_match = None
            header_level = 0

            for pattern, level in self.header_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    header_match = match.group(1)
                    header_level = level
                    break

            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = "\n".join(current_content)
                    sections.append(current_section)

                # Start new section
                current_section = Section(
                    title=header_match,
                    level=header_level,
                    content="",
                    subsections=[]
                )
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            current_section.content = "\n".join(current_content)
            sections.append(current_section)
        elif current_content:
            # No headers found
            sections.append(Section(
                title="",
                level=0,
                content="\n".join(current_content),
                subsections=[]
            ))

        return sections

    def section_to_chunks(self, section: Section) -> list[dict]:
        chunks = []

        # Build section text
        if self.include_title_in_chunk and section.title:
            section_text = f"{section.title}\n\n{section.content}"
        else:
            section_text = section.content

        # If section fits in one chunk
        if len(section_text) <= self.max_chunk_size:
            if len(section_text) >= self.min_chunk_size:
                chunks.append({
                    "content": section_text.strip(),
                    "metadata": {
                        "section_title": section.title,
                        "section_level": section.level
                    }
                })
        else:
            # Split large section
            sub_chunks = self.split_section(section_text, section)
            chunks.extend(sub_chunks)

        return chunks

    def split_section(self, text: str, section: Section) -> list[dict]:
        # Use paragraph-based splitting within section
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": {
                            "section_title": section.title,
                            "section_level": section.level
                        }
                    })
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    "section_title": section.title,
                    "section_level": section.level
                }
            })

        return chunks
```

| Pros | Cons |
|------|------|
| Preserves document structure | Requires parsing |
| Natural section boundaries | Format-dependent |
| Better context preservation | Variable chunk sizes |
| Rich metadata | More complex setup |

**When to Use:**
- Structured documents (manuals, reports)
- When document hierarchy matters
- Quality-critical applications

---

### 3.5 Agentic Chunking

Use LLM to determine optimal chunk boundaries.

```python
class AgenticChunker:
    """Use LLM to create semantically coherent chunks."""

    def __init__(self, llm_client, target_size: int = 500):
        self.llm = llm_client
        self.target_size = target_size

    async def chunk(self, text: str) -> list[dict]:
        # Step 1: Extract propositions
        propositions = await self.extract_propositions(text)

        # Step 2: Group propositions into chunks
        chunks = await self.group_propositions(propositions)

        return chunks

    async def extract_propositions(self, text: str) -> list[str]:
        """Extract atomic facts/propositions from text."""
        prompt = f"""
        Extract all factual propositions from this text.
        Each proposition should be:
        - Self-contained (understandable without context)
        - Atomic (one fact per proposition)
        - Decontextualized (resolve pronouns, add context)

        Text:
        {text}

        Propositions (one per line):
        """

        response = await self.llm.generate(prompt)
        return [p.strip() for p in response.split("\n") if p.strip()]

    async def group_propositions(self, propositions: list[str]) -> list[dict]:
        """Group related propositions into chunks."""
        chunks = []
        current_chunk = {
            "title": "",
            "propositions": [],
            "content": ""
        }

        for prop in propositions:
            # Check if proposition belongs to current chunk
            if current_chunk["propositions"]:
                belongs = await self.check_belongs(prop, current_chunk)

                if not belongs or len(current_chunk["content"]) > self.target_size:
                    # Finalize current chunk
                    current_chunk["content"] = " ".join(current_chunk["propositions"])
                    if current_chunk["content"]:
                        chunks.append(current_chunk)

                    # Start new chunk
                    current_chunk = {
                        "title": await self.generate_title([prop]),
                        "propositions": [prop],
                        "content": ""
                    }
                else:
                    current_chunk["propositions"].append(prop)
            else:
                current_chunk["propositions"].append(prop)
                current_chunk["title"] = await self.generate_title([prop])

        # Add last chunk
        if current_chunk["propositions"]:
            current_chunk["content"] = " ".join(current_chunk["propositions"])
            chunks.append(current_chunk)

        return chunks

    async def check_belongs(self, proposition: str, chunk: dict) -> bool:
        """Check if proposition belongs to current chunk."""
        prompt = f"""
        Does this new proposition belong in the same chunk as the existing content?

        Existing chunk topic: {chunk['title']}
        Existing propositions: {chunk['propositions'][:3]}

        New proposition: {proposition}

        Answer (yes/no):
        """

        response = await self.llm.generate(prompt)
        return "yes" in response.lower()

    async def generate_title(self, propositions: list[str]) -> str:
        """Generate a title for chunk."""
        prompt = f"""
        Generate a short title (3-5 words) for these propositions:
        {propositions}

        Title:
        """

        return await self.llm.generate(prompt)
```

| Pros | Cons |
|------|------|
| Optimal semantic chunks | Expensive (many LLM calls) |
| Self-contained chunks | Slow processing |
| Intelligent grouping | Non-deterministic |
| Handles complex content | Overkill for simple docs |

**When to Use:**
- Highest quality requirements
- Complex, unstructured content
- When cost is not a concern

[↑ Back to Top](#table-of-contents)

---

## 4. Overlap Strategies

### 4.1 Why Overlap?

```
Without overlap:
┌─────────────┐┌─────────────┐┌─────────────┐
│   Chunk 1   ││   Chunk 2   ││   Chunk 3   │
│             ││             ││             │
│     ...the  ││process was  ││essential for│
│    process  ││essential... ││    ...      │
└─────────────┘└─────────────┘└─────────────┘
              ↑              ↑
         Lost context!  Lost context!

With overlap:
┌─────────────┐
│   Chunk 1   │
│             │
│  ...the     │
│  process    ├──┐
└─────────────┘  │ Overlap
┌────────────────┤
│   Chunk 2      │
│                │
│ the process was│
│ essential for  ├──┐
└────────────────┘  │ Overlap
    ┌───────────────┤
    │   Chunk 3     │
    │               │
    │ essential for │
    │ ...           │
    └───────────────┘
```

### 4.2 Overlap Guidelines

| Overlap % | Token Overlap | Use Case |
|-----------|---------------|----------|
| **0%** | 0 | Independent sections (forms, FAQs) |
| **10%** | 50 tokens | Standard documents |
| **15-20%** | 75-100 tokens | Technical content (recommended default) |
| **25-30%** | 125-150 tokens | Dense information, legal |
| **50%** | Half chunk | Critical content, sliding window |

### 4.3 Overlap Implementation

```python
class OverlapChunker:
    """Chunking with configurable overlap strategies."""

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 100,
        overlap_strategy: str = "fixed"  # "fixed", "sentence", "semantic"
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.overlap_strategy = overlap_strategy

    def chunk(self, text: str) -> list[str]:
        if self.overlap_strategy == "fixed":
            return self.fixed_overlap(text)
        elif self.overlap_strategy == "sentence":
            return self.sentence_overlap(text)
        elif self.overlap_strategy == "semantic":
            return self.semantic_overlap(text)

    def fixed_overlap(self, text: str) -> list[str]:
        """Simple character-based overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move forward, minus overlap
            start = end - self.overlap_size

        return chunks

    def sentence_overlap(self, text: str) -> list[str]:
        """Overlap at sentence boundaries."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0
        overlap_sentences = []

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Calculate overlap (last N sentences)
                overlap_length = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    overlap_length += len(s)
                    overlap_sentences.insert(0, s)
                    if overlap_length >= self.overlap_size:
                        break

                # Start new chunk with overlap
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
```

### 4.4 Sliding Window

Maximum overlap approach for critical content:

```python
def sliding_window_chunks(text: str, window_size: int = 512, stride: int = 256) -> list[str]:
    """
    Create overlapping chunks with sliding window.

    With stride = window_size/2, you get 50% overlap.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += stride

    return chunks

# Example: 512 token windows with 50% overlap
chunks = sliding_window_chunks(document, window_size=512, stride=256)
```

[↑ Back to Top](#table-of-contents)

---

## 5. Advanced Chunking

### 5.1 Parent-Child Chunking

Store large parent chunks, retrieve by small child chunks.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

class ParentChildChunker:
    """Create parent chunks for context, child chunks for retrieval."""

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        child_overlap: int = 50
    ):
        self.parent_size = parent_chunk_size
        self.child_size = child_chunk_size
        self.child_overlap = child_overlap

    def chunk(self, text: str) -> dict:
        """
        Returns:
        {
            "parents": [...],  # Large chunks for context
            "children": [...], # Small chunks for retrieval
            "child_to_parent": {...}  # Mapping
        }
        """
        # Create parent chunks
        parents = self.create_parents(text)

        # Create child chunks from each parent
        children = []
        child_to_parent = {}

        for parent_idx, parent in enumerate(parents):
            parent_children = self.create_children(parent["content"])

            for child in parent_children:
                child_id = len(children)
                child["parent_id"] = parent_idx
                children.append(child)
                child_to_parent[child_id] = parent_idx

        return {
            "parents": parents,
            "children": children,
            "child_to_parent": child_to_parent
        }

    def create_parents(self, text: str) -> list[dict]:
        # Simple paragraph-based parent chunking
        paragraphs = text.split("\n\n")
        parents = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < self.parent_size:
                current += para + "\n\n"
            else:
                if current:
                    parents.append({"content": current.strip(), "id": len(parents)})
                current = para + "\n\n"

        if current:
            parents.append({"content": current.strip(), "id": len(parents)})

        return parents

    def create_children(self, parent_text: str) -> list[dict]:
        # Create smaller chunks from parent
        children = []
        start = 0

        while start < len(parent_text):
            end = start + self.child_size
            child_text = parent_text[start:end]
            children.append({"content": child_text})
            start = end - self.child_overlap

        return children
```

**How it works in retrieval:**

```python
class ParentChildRetriever:
    def __init__(self, vector_store, parent_store):
        self.vector_store = vector_store
        self.parent_store = parent_store

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        # Search child chunks
        child_results = self.vector_store.similarity_search(query, k=k)

        # Get unique parent chunks
        parent_ids = set()
        for child in child_results:
            parent_ids.add(child.metadata["parent_id"])

        # Return parent content (richer context)
        parents = []
        for parent_id in parent_ids:
            parent = self.parent_store.get(parent_id)
            parents.append(parent)

        return parents
```

| Pros | Cons |
|------|------|
| Best of both worlds | More storage |
| Precise retrieval | Complex setup |
| Rich context | Index management |

---

### 5.2 Multi-Vector Chunking

Create multiple vector representations per chunk.

```python
class MultiVectorChunker:
    """Generate multiple vectors per chunk for better retrieval."""

    def __init__(self, llm_client, embedding_model):
        self.llm = llm_client
        self.embedder = embedding_model

    async def process_chunk(self, chunk: str) -> dict:
        """Create multiple vectors for a single chunk."""

        # Original embedding
        original_embedding = self.embedder.encode(chunk)

        # Summary embedding
        summary = await self.generate_summary(chunk)
        summary_embedding = self.embedder.encode(summary)

        # Question embeddings
        questions = await self.generate_questions(chunk)
        question_embeddings = [self.embedder.encode(q) for q in questions]

        return {
            "content": chunk,
            "vectors": {
                "original": original_embedding,
                "summary": summary_embedding,
                "questions": question_embeddings
            },
            "metadata": {
                "summary": summary,
                "questions": questions
            }
        }

    async def generate_summary(self, chunk: str) -> str:
        prompt = f"Summarize this text in 1-2 sentences:\n{chunk}"
        return await self.llm.generate(prompt)

    async def generate_questions(self, chunk: str, n: int = 3) -> list[str]:
        prompt = f"""
        Generate {n} questions that this text answers:

        Text: {chunk}

        Questions:
        """
        response = await self.llm.generate(prompt)
        return [q.strip() for q in response.split("\n") if q.strip()]
```

---

### 5.3 Code-Aware Chunking

Special handling for source code.

```python
import ast
from tree_sitter import Language, Parser

class CodeAwareChunker:
    """Chunk code files by syntactic units."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk_python(self, code: str) -> list[dict]:
        """Chunk Python code by functions and classes."""
        chunks = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to simple chunking
            return [{"content": code, "type": "raw"}]

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self.extract_function(code, node)
                chunks.append({
                    "content": chunk,
                    "type": "function",
                    "name": node.name,
                    "docstring": ast.get_docstring(node)
                })

            elif isinstance(node, ast.ClassDef):
                chunk = self.extract_class(code, node)
                chunks.append({
                    "content": chunk,
                    "type": "class",
                    "name": node.name,
                    "docstring": ast.get_docstring(node)
                })

        # Add module-level content
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunks.insert(0, {
                "content": module_docstring,
                "type": "module_docstring"
            })

        return chunks

    def extract_function(self, code: str, node) -> str:
        lines = code.split("\n")
        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(lines[start:end])

    def extract_class(self, code: str, node) -> str:
        lines = code.split("\n")
        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(lines[start:end])
```

[↑ Back to Top](#table-of-contents)

---

## 6. Problem Statements & Solutions

### Problem 1: Incomplete Answers

**Symptoms:**
- Answers missing crucial information
- "The document mentions X but doesn't explain..."
- Context seems cut off

**Root Cause:** Chunks are too small

**Solution:**

```python
# Increase chunk size
config = {
    "chunk_size": 768,  # Increased from 256
    "chunk_overlap": 150,  # Increased overlap
}

# Or use parent-child retrieval
parent_child_retriever = ParentDocumentRetriever(
    parent_chunk_size=2000,
    child_chunk_size=400
)
```

---

### Problem 2: Irrelevant Results Retrieved

**Symptoms:**
- Retrieved chunks don't match query
- Too much noise in results
- Precision is low

**Root Cause:** Chunks are too large, diluting relevance

**Solution:**

```python
# Decrease chunk size
config = {
    "chunk_size": 256,  # Decreased from 512
    "chunk_overlap": 50,
}

# Add re-ranking
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve_with_rerank(query: str, k: int = 5):
    # Get more candidates
    candidates = vector_store.similarity_search(query, k=20)

    # Re-rank
    pairs = [(query, doc.content) for doc in candidates]
    scores = reranker.predict(pairs)

    # Return top-k after re-ranking
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]
```

---

### Problem 3: Split Sentences/Paragraphs

**Symptoms:**
- Chunks end mid-sentence
- Context feels broken
- Awkward text fragments

**Root Cause:** Using character-based chunking without boundary awareness

**Solution:**

```python
class BoundaryAwareChunker:
    """Never split mid-sentence."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save chunk
                chunks.append(" ".join(current_chunk))

                # Calculate overlap (include last N sentences)
                overlap_sentences = self.get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        """Get sentences for overlap."""
        overlap = []
        length = 0

        for sentence in reversed(sentences):
            if length + len(sentence) <= self.overlap:
                overlap.insert(0, sentence)
                length += len(sentence)
            else:
                break

        return overlap
```

---

### Problem 4: Lost Table Data

**Symptoms:**
- Table content is garbled
- Row/column relationships lost
- Numbers without context

**Root Cause:** Tables need special handling

**Solution:**

```python
class TableAwareChunker:
    """Preserve table structure in chunks."""

    def chunk_with_tables(self, document: dict) -> list[dict]:
        chunks = []

        for element in document["elements"]:
            if element["type"] == "table":
                # Convert table to structured text
                table_chunk = self.table_to_chunk(element["data"])
                chunks.append(table_chunk)
            else:
                # Regular text chunking
                text_chunks = self.chunk_text(element["content"])
                chunks.extend(text_chunks)

        return chunks

    def table_to_chunk(self, table_data: list[list]) -> dict:
        """Convert table to retrievable chunk."""

        # Option 1: Markdown format
        markdown = self.table_to_markdown(table_data)

        # Option 2: Row-by-row description
        description = self.table_to_description(table_data)

        return {
            "content": f"{markdown}\n\n{description}",
            "type": "table",
            "raw_data": table_data
        }

    def table_to_markdown(self, data: list[list]) -> str:
        if not data:
            return ""

        headers = data[0]
        rows = data[1:]

        # Header row
        md = "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        # Data rows
        for row in rows:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"

        return md

    def table_to_description(self, data: list[list]) -> str:
        if not data:
            return ""

        headers = data[0]
        rows = data[1:]
        descriptions = []

        for row in rows:
            desc = "; ".join(f"{headers[i]}: {row[i]}" for i in range(len(headers)))
            descriptions.append(desc)

        return "\n".join(descriptions)
```

---

### Problem 5: Code Chunks Don't Make Sense

**Symptoms:**
- Functions split in the middle
- Lost import context
- Incomplete code blocks

**Root Cause:** Code needs syntax-aware chunking

**Solution:**

```python
class CodeChunker:
    """Chunk code by syntactic units."""

    def chunk(self, code: str, language: str = "python") -> list[dict]:
        if language == "python":
            return self.chunk_python(code)
        else:
            return self.chunk_generic(code)

    def chunk_python(self, code: str) -> list[dict]:
        import ast

        chunks = []
        lines = code.split("\n")

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self.chunk_generic(code)

        # Collect imports at the top
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_line = lines[node.lineno - 1]
                imports.append(import_line)

        import_context = "\n".join(imports)

        # Extract functions and classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno

                func_code = "\n".join(lines[start:end])

                # Include imports for context
                full_chunk = f"# Imports\n{import_context}\n\n# Code\n{func_code}"

                chunks.append({
                    "content": full_chunk,
                    "type": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
                    "name": node.name
                })

        return chunks

    def chunk_generic(self, code: str) -> list[dict]:
        """Fallback: chunk by blank lines."""
        blocks = code.split("\n\n")
        return [{"content": block, "type": "code_block"} for block in blocks if block.strip()]
```

[↑ Back to Top](#table-of-contents)

---

## 7. Trade-offs

### Size Trade-offs

| Factor | Small Chunks | Large Chunks |
|--------|--------------|--------------|
| **Retrieval Precision** | ✅ High | ❌ Low |
| **Context Completeness** | ❌ Low | ✅ High |
| **Embedding Quality** | ✅ Better | ⚠️ May degrade |
| **Search Speed** | ❌ More to search | ✅ Fewer chunks |
| **Storage Cost** | ❌ More embeddings | ✅ Fewer embeddings |
| **Context Window Usage** | ✅ Efficient | ❌ May waste tokens |

### Method Trade-offs

| Method | Quality | Speed | Complexity | Cost |
|--------|---------|-------|------------|------|
| **Fixed-size** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Free |
| **Recursive** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Free |
| **Semantic** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | $ (embeddings) |
| **Document-aware** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Free |
| **Agentic** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | $$$ (LLM calls) |

### Overlap Trade-offs

| Overlap | Storage | Recall | Redundancy |
|---------|---------|--------|------------|
| **0%** | 1x | Lower | None |
| **20%** | 1.25x | Moderate | Low |
| **50%** | 2x | High | Moderate |

[↑ Back to Top](#table-of-contents)

---

## 8. Cost-Effective Solutions

### Free Chunking Stack

```python
# Best free chunking setup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Works well for 90% of cases
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)
```

### Cost Comparison

| Method | Processing Cost | Storage Impact |
|--------|-----------------|----------------|
| **Fixed/Recursive** | $0 | 1x |
| **Semantic** | ~$0.01 per 1000 chunks | 1x |
| **Agentic** | ~$0.50 per document | 1x |
| **Parent-Child** | $0 | 1.5-2x |
| **Multi-Vector** | ~$0.10 per document | 3-5x |

### Optimization Tips

1. **Start Simple, Iterate**
   - Begin with recursive character splitting
   - Add complexity only if needed

2. **Batch Semantic Chunking**
   - Process embeddings in batches
   - Use local embedding models

3. **Cache Chunk Results**
   - Don't re-chunk unchanged documents
   - Store chunk hashes

4. **Use Parent-Child Selectively**
   - Only for documents needing rich context
   - Not needed for FAQs

[↑ Back to Top](#table-of-contents)

---

## 9. Best Practices

### DO's

1. **Test Multiple Chunk Sizes**
   ```python
   # A/B test chunk sizes
   sizes_to_test = [256, 512, 768, 1024]

   for size in sizes_to_test:
       chunks = create_chunks(document, size=size)
       quality = evaluate_retrieval_quality(chunks, test_queries)
       print(f"Size {size}: Quality {quality}")
   ```

2. **Include Rich Metadata**
   ```python
   chunk = {
       "content": text,
       "metadata": {
           "source": "handbook.pdf",
           "page": 15,
           "section": "HR Policies",
           "chunk_index": 42,
           "total_chunks": 150,
           "created_at": "2024-01-15"
       }
   }
   ```

3. **Validate Chunk Quality**
   ```python
   def validate_chunk(chunk: str) -> bool:
       # Minimum length
       if len(chunk.split()) < 20:
           return False

       # Has sentence structure
       if "." not in chunk and "?" not in chunk:
           return False

       # Not mostly whitespace
       if len(chunk.strip()) / len(chunk) < 0.8:
           return False

       return True
   ```

4. **Handle Edge Cases**
   - Empty documents
   - Single-line documents
   - Documents with only tables
   - Very long paragraphs

### DON'Ts

1. **Don't Use One Size for Everything**
   - Different content needs different chunking

2. **Don't Ignore Overlap**
   - Always use some overlap (10-20%)

3. **Don't Chunk Without Context**
   - Add headers/section info to chunks

4. **Don't Over-Engineer Initially**
   - Start simple, optimize based on data

[↑ Back to Top](#table-of-contents)

---

## 10. Quick Reference

### Chunk Size Cheat Sheet

```
Document Type → Chunk Size (tokens)
─────────────────────────────────────
FAQ / Q&A          →  128-256
Support Tickets    →  128-256
Knowledge Base     →  256-512
Technical Docs     →  512-768
Legal Documents    →  512-1024
Research Papers    →  768-1024
Code Files         →  256-512
```

### Method Selection

```
Need → Method
─────────────────────────────────────
Quick prototype    →  Fixed-size
General documents  →  Recursive
Quality-critical   →  Semantic
Structured docs    →  Document-aware
Maximum quality    →  Agentic + Parent-Child
```

### Overlap Guidelines

```
Content Type → Overlap %
─────────────────────────────────────
Independent (FAQ)    →  0-10%
Standard            →  15-20%
Technical           →  20-25%
Dense/Critical      →  25-50%
```

### Code Example: Production-Ready Chunker

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_production_chunker(doc_type: str) -> RecursiveCharacterTextSplitter:
    """Create optimized chunker based on document type."""

    configs = {
        "faq": {"chunk_size": 300, "chunk_overlap": 30},
        "knowledge_base": {"chunk_size": 500, "chunk_overlap": 100},
        "technical": {"chunk_size": 800, "chunk_overlap": 150},
        "legal": {"chunk_size": 1000, "chunk_overlap": 200},
        "default": {"chunk_size": 500, "chunk_overlap": 100},
    }

    config = configs.get(doc_type, configs["default"])

    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Data Cleanup](./01-data-cleanup.md) | [Main Guide](../README.md) | [Embeddings →](./03-embeddings.md) |

---

*Last updated: 2024*
