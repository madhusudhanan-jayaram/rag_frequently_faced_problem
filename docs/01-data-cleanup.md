# Data Cleanup & Preprocessing for RAG

Complete guide to preparing documents for Retrieval-Augmented Generation systems.

[← Back to Main Guide](../README.md)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Document Formats & Parsing](#2-document-formats--parsing)
- [3. Implementation Strategies](#3-implementation-strategies)
- [4. Preprocessing Pipeline](#4-preprocessing-pipeline)
- [5. Problem Statements & Solutions](#5-problem-statements--solutions)
- [6. Trade-offs](#6-trade-offs)
- [7. Cost-Effective Solutions](#7-cost-effective-solutions)
- [8. Best Practices](#8-best-practices)
- [9. Quick Reference](#9-quick-reference)

---

## 1. Overview

### Why Data Cleanup Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                    "Garbage In, Garbage Out"                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Poor Data Quality          →    Poor Retrieval Quality         │
│  ─────────────────               ──────────────────────         │
│  • Broken text              →    • Irrelevant results           │
│  • Missing content          →    • Incomplete answers           │
│  • Noise & artifacts        →    • Hallucinations               │
│  • Inconsistent formats     →    • Unpredictable behavior       │
│                                                                  │
│  Good Data Quality          →    Good Retrieval Quality         │
│  ─────────────────               ──────────────────────         │
│  • Clean text               →    • Relevant results             │
│  • Complete extraction      →    • Comprehensive answers        │
│  • Structured metadata      →    • Accurate responses           │
│  • Normalized formats       →    • Consistent behavior          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Cleanup Pipeline Overview

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Ingest  │──▶│  Parse   │──▶│  Clean   │──▶│  Enrich  │──▶│ Validate │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
  Load raw      Extract       Remove noise    Add metadata   Quality
  documents     content       & normalize     & structure    check
```

[↑ Back to Top](#table-of-contents)

---

## 2. Document Formats & Parsing

### 2.1 Supported Formats Overview

| Format | Extension | Complexity | Common Issues |
|--------|-----------|------------|---------------|
| **Plain Text** | .txt | Low | Encoding issues |
| **Markdown** | .md | Low | Nested structures |
| **HTML** | .html | Medium | Boilerplate, scripts |
| **PDF** | .pdf | High | Scanned images, complex layouts |
| **Word** | .docx | Medium | Styles, tracked changes |
| **Excel** | .xlsx | Medium | Multiple sheets, formulas |
| **PowerPoint** | .pptx | Medium | Speaker notes, animations |
| **Email** | .eml, .msg | Medium | Attachments, threads |
| **Images** | .png, .jpg | High | OCR required |

### 2.2 Parser Comparison

| Parser | Formats | Strengths | Weaknesses | Cost |
|--------|---------|-----------|------------|------|
| **Unstructured.io** | All major | Best all-rounder, structure-aware | Setup complexity | Free/Paid |
| **PyPDF2** | PDF | Simple, fast | No OCR, basic extraction | Free |
| **pdfplumber** | PDF | Good table extraction | No OCR | Free |
| **PyMuPDF (fitz)** | PDF | Fast, feature-rich | Learning curve | Free |
| **python-docx** | DOCX | Native Word support | Limited formatting | Free |
| **BeautifulSoup** | HTML | Flexible, popular | Manual work needed | Free |
| **Trafilatura** | HTML/Web | Excellent content extraction | Web-focused | Free |
| **Azure Form Recognizer** | All + OCR | Enterprise OCR, high quality | Cost | $$$ |
| **AWS Textract** | All + OCR | Good table/form extraction | Cost | $$$ |
| **Google Document AI** | All + OCR | Accurate, structured output | Cost | $$$ |

### 2.3 Format-Specific Parsing

#### PDF Parsing

```python
# Option 1: Simple PDF extraction
import pdfplumber

def parse_pdf_simple(file_path: str) -> str:
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

# Option 2: Structure-aware extraction with tables
def parse_pdf_with_tables(file_path: str) -> dict:
    content = {"text": [], "tables": []}

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                content["text"].append({
                    "page": page_num + 1,
                    "content": text
                })

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                content["tables"].append({
                    "page": page_num + 1,
                    "data": table
                })

    return content

# Option 3: OCR for scanned PDFs
import pytesseract
from pdf2image import convert_from_path

def parse_scanned_pdf(file_path: str) -> str:
    images = convert_from_path(file_path)
    text = []

    for image in images:
        text.append(pytesseract.image_to_string(image))

    return "\n".join(text)
```

#### HTML Parsing

```python
from trafilatura import extract
from bs4 import BeautifulSoup
import requests

# Option 1: Trafilatura (recommended for web content)
def parse_html_trafilatura(url_or_html: str) -> str:
    if url_or_html.startswith("http"):
        downloaded = trafilatura.fetch_url(url_or_html)
        return extract(downloaded)
    return extract(url_or_html)

# Option 2: BeautifulSoup (more control)
def parse_html_bs4(html: str, remove_selectors: list = None) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    remove_selectors = remove_selectors or [
        "script", "style", "nav", "footer", "header", "aside"
    ]
    for selector in remove_selectors:
        for element in soup.select(selector):
            element.decompose()

    # Get text
    text = soup.get_text(separator="\n", strip=True)
    return text
```

#### Word/Excel Parsing

```python
from docx import Document
import pandas as pd

def parse_docx(file_path: str) -> dict:
    doc = Document(file_path)
    content = {
        "paragraphs": [],
        "tables": [],
        "headers": []
    }

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            content["headers"].append({
                "level": para.style.name,
                "text": para.text
            })
        content["paragraphs"].append(para.text)

    for table in doc.tables:
        table_data = []
        for row in table.rows:
            table_data.append([cell.text for cell in row.cells])
        content["tables"].append(table_data)

    return content

def parse_excel(file_path: str) -> dict:
    # Read all sheets
    excel_file = pd.ExcelFile(file_path)
    content = {}

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        content[sheet_name] = df.to_string()

    return content
```

[↑ Back to Top](#table-of-contents)

---

## 3. Implementation Strategies

### 3.1 Unified Parsing Pipeline

```python
from pathlib import Path
from typing import Union
from dataclasses import dataclass

@dataclass
class ParsedDocument:
    content: str
    metadata: dict
    tables: list
    images: list
    quality_score: float

class UnifiedDocumentParser:
    """Parse any document format into a standardized structure."""

    def __init__(self, ocr_enabled: bool = False):
        self.ocr_enabled = ocr_enabled
        self.parsers = {
            ".pdf": self.parse_pdf,
            ".docx": self.parse_docx,
            ".doc": self.parse_doc,
            ".xlsx": self.parse_excel,
            ".xls": self.parse_excel,
            ".html": self.parse_html,
            ".htm": self.parse_html,
            ".txt": self.parse_text,
            ".md": self.parse_markdown,
            ".pptx": self.parse_pptx,
        }

    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.parsers:
            raise ValueError(f"Unsupported format: {extension}")

        # Parse content
        result = self.parsers[extension](file_path)

        # Extract metadata
        metadata = self.extract_metadata(file_path, result)

        # Calculate quality score
        quality_score = self.calculate_quality(result)

        return ParsedDocument(
            content=result["text"],
            metadata=metadata,
            tables=result.get("tables", []),
            images=result.get("images", []),
            quality_score=quality_score
        )

    def extract_metadata(self, file_path: Path, content: dict) -> dict:
        import os
        stat = os.stat(file_path)

        return {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "word_count": len(content["text"].split()),
            "has_tables": len(content.get("tables", [])) > 0,
            "has_images": len(content.get("images", [])) > 0,
        }

    def calculate_quality(self, content: dict) -> float:
        """Score from 0-1 based on extraction quality."""
        text = content["text"]

        if not text:
            return 0.0

        score = 1.0

        # Penalize short extractions
        if len(text) < 100:
            score -= 0.3

        # Penalize high ratio of special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and c != " ") / len(text)
        if special_ratio > 0.3:
            score -= 0.2

        # Penalize lack of sentence structure
        if "." not in text and "!" not in text and "?" not in text:
            score -= 0.2

        return max(0.0, score)
```

### 3.2 Using Unstructured.io (Recommended)

```python
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Table

def parse_with_unstructured(file_path: str, strategy: str = "auto") -> dict:
    """
    Parse document using unstructured.io

    Strategies:
    - "auto": Automatic detection
    - "fast": Quick extraction, lower quality
    - "hi_res": High resolution, uses OCR/vision models
    - "ocr_only": Force OCR
    """

    elements = partition(
        filename=file_path,
        strategy=strategy,
        include_metadata=True,
        include_page_breaks=True
    )

    result = {
        "text": [],
        "tables": [],
        "titles": [],
        "metadata": []
    }

    for element in elements:
        if isinstance(element, Title):
            result["titles"].append(element.text)
        elif isinstance(element, Table):
            result["tables"].append(element.text)
        elif isinstance(element, NarrativeText):
            result["text"].append(element.text)

        # Collect metadata
        if hasattr(element, "metadata"):
            result["metadata"].append(element.metadata.to_dict())

    return {
        "text": "\n\n".join(result["text"]),
        "tables": result["tables"],
        "titles": result["titles"],
        "metadata": result["metadata"]
    }

# For complex PDFs with tables
def parse_complex_pdf(file_path: str) -> dict:
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
    )

    return process_elements(elements)
```

[↑ Back to Top](#table-of-contents)

---

## 4. Preprocessing Pipeline

### 4.1 Text Cleaning

```python
import re
import unicodedata
from ftfy import fix_text

class TextCleaner:
    """Clean and normalize text for RAG."""

    def __init__(self):
        self.patterns = {
            "multiple_spaces": re.compile(r" +"),
            "multiple_newlines": re.compile(r"\n{3,}"),
            "urls": re.compile(r"https?://\S+"),
            "emails": re.compile(r"\S+@\S+\.\S+"),
            "phone_numbers": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "special_chars": re.compile(r"[^\w\s\.\,\!\?\-\:\;\'\"\(\)]"),
        }

    def clean(self, text: str, config: dict = None) -> str:
        """
        Clean text with configurable options.

        Config options:
        - fix_encoding: Fix Unicode issues (default: True)
        - normalize_whitespace: Normalize spaces/newlines (default: True)
        - remove_urls: Remove URLs (default: False)
        - remove_emails: Remove email addresses (default: False)
        - remove_phone_numbers: Remove phone numbers (default: False)
        - lowercase: Convert to lowercase (default: False)
        - remove_special_chars: Remove special characters (default: False)
        """
        config = config or {}

        # Fix encoding issues
        if config.get("fix_encoding", True):
            text = fix_text(text)
            text = unicodedata.normalize("NFKC", text)

        # Normalize whitespace
        if config.get("normalize_whitespace", True):
            text = self.patterns["multiple_spaces"].sub(" ", text)
            text = self.patterns["multiple_newlines"].sub("\n\n", text)
            text = text.strip()

        # Remove URLs
        if config.get("remove_urls", False):
            text = self.patterns["urls"].sub("[URL]", text)

        # Remove emails
        if config.get("remove_emails", False):
            text = self.patterns["emails"].sub("[EMAIL]", text)

        # Remove phone numbers
        if config.get("remove_phone_numbers", False):
            text = self.patterns["phone_numbers"].sub("[PHONE]", text)

        # Lowercase
        if config.get("lowercase", False):
            text = text.lower()

        # Remove special characters
        if config.get("remove_special_chars", False):
            text = self.patterns["special_chars"].sub("", text)

        return text

    def clean_for_embedding(self, text: str) -> str:
        """Optimized cleaning for embedding generation."""
        return self.clean(text, {
            "fix_encoding": True,
            "normalize_whitespace": True,
            "remove_urls": False,  # URLs can be meaningful
            "lowercase": False,     # Case can be meaningful
        })
```

### 4.2 Noise Removal

```python
class NoiseRemover:
    """Remove common noise patterns from documents."""

    def __init__(self):
        self.header_footer_patterns = [
            re.compile(r"^Page \d+ of \d+$", re.MULTILINE),
            re.compile(r"^\d+$", re.MULTILINE),  # Page numbers
            re.compile(r"^(Confidential|Draft|Internal Use Only).*$", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^Copyright ©.*$", re.MULTILINE),
        ]

        self.boilerplate_patterns = [
            re.compile(r"^(Table of Contents|Index|Appendix)$", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^\[?\d+\]?\s*$", re.MULTILINE),  # Reference numbers
        ]

    def remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns."""
        for pattern in self.header_footer_patterns:
            text = pattern.sub("", text)
        return text

    def remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate content."""
        for pattern in self.boilerplate_patterns:
            text = pattern.sub("", text)
        return text

    def remove_repetitive_content(self, text: str, threshold: int = 3) -> str:
        """Remove lines that repeat more than threshold times."""
        lines = text.split("\n")
        line_counts = {}

        for line in lines:
            line_clean = line.strip()
            if line_clean:
                line_counts[line_clean] = line_counts.get(line_clean, 0) + 1

        # Filter out repetitive lines
        result = []
        for line in lines:
            line_clean = line.strip()
            if not line_clean or line_counts.get(line_clean, 0) <= threshold:
                result.append(line)

        return "\n".join(result)

    def clean(self, text: str) -> str:
        """Apply all noise removal."""
        text = self.remove_headers_footers(text)
        text = self.remove_boilerplate(text)
        text = self.remove_repetitive_content(text)
        return text
```

### 4.3 Metadata Enrichment

```python
from datetime import datetime
import hashlib
from typing import Optional
import spacy

class MetadataEnricher:
    """Enrich documents with metadata for better retrieval."""

    def __init__(self, nlp_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(nlp_model)

    def enrich(self, text: str, source_metadata: dict = None) -> dict:
        """Generate comprehensive metadata for a document."""
        source_metadata = source_metadata or {}

        # Basic stats
        doc = self.nlp(text[:100000])  # Limit for performance

        metadata = {
            # Source info
            "source": source_metadata.get("source", "unknown"),
            "filename": source_metadata.get("filename"),
            "file_type": source_metadata.get("file_type"),

            # Content hash for deduplication
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],

            # Statistics
            "char_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(list(doc.sents)),

            # Language detection
            "language": self.detect_language(text),

            # Named entities
            "entities": self.extract_entities(doc),

            # Topics/keywords
            "keywords": self.extract_keywords(doc),

            # Timestamps
            "processed_at": datetime.utcnow().isoformat(),
            "source_date": source_metadata.get("date"),
        }

        return metadata

    def detect_language(self, text: str) -> str:
        """Detect document language."""
        from langdetect import detect
        try:
            return detect(text[:1000])
        except:
            return "unknown"

    def extract_entities(self, doc) -> dict:
        """Extract named entities."""
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        return entities

    def extract_keywords(self, doc, top_n: int = 10) -> list:
        """Extract keywords using noun chunks."""
        keywords = []
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2:
                keywords.append(chunk.text.lower())

        # Return most common
        from collections import Counter
        return [kw for kw, _ in Counter(keywords).most_common(top_n)]
```

### 4.4 Quality Validation

```python
@dataclass
class QualityReport:
    passed: bool
    score: float
    issues: list
    suggestions: list

class QualityValidator:
    """Validate document quality for RAG."""

    def __init__(self, min_score: float = 0.6):
        self.min_score = min_score

    def validate(self, text: str, metadata: dict = None) -> QualityReport:
        """Run all quality checks."""
        issues = []
        suggestions = []
        scores = []

        # Check 1: Minimum length
        length_score = self.check_length(text)
        scores.append(length_score)
        if length_score < 1.0:
            issues.append("Document is very short")
            suggestions.append("Verify extraction completed successfully")

        # Check 2: Encoding quality
        encoding_score = self.check_encoding(text)
        scores.append(encoding_score)
        if encoding_score < 1.0:
            issues.append("Encoding issues detected")
            suggestions.append("Re-process with encoding fix enabled")

        # Check 3: Content density
        density_score = self.check_content_density(text)
        scores.append(density_score)
        if density_score < 0.8:
            issues.append("Low content density (too much whitespace/noise)")
            suggestions.append("Apply noise removal")

        # Check 4: Language coherence
        coherence_score = self.check_coherence(text)
        scores.append(coherence_score)
        if coherence_score < 0.8:
            issues.append("Text may be garbled or OCR errors present")
            suggestions.append("Consider re-OCR with better settings")

        # Calculate overall score
        overall_score = sum(scores) / len(scores)

        return QualityReport(
            passed=overall_score >= self.min_score,
            score=overall_score,
            issues=issues,
            suggestions=suggestions
        )

    def check_length(self, text: str, min_chars: int = 100) -> float:
        """Check if document meets minimum length."""
        if len(text) >= min_chars:
            return 1.0
        return len(text) / min_chars

    def check_encoding(self, text: str) -> float:
        """Check for encoding issues."""
        # Look for common encoding artifacts
        bad_patterns = ["�", "Ã", "â€", "Â"]
        bad_count = sum(text.count(p) for p in bad_patterns)

        if bad_count == 0:
            return 1.0

        ratio = bad_count / len(text)
        return max(0, 1 - ratio * 100)

    def check_content_density(self, text: str) -> float:
        """Check ratio of actual content to whitespace."""
        if not text:
            return 0.0

        content_chars = sum(1 for c in text if c.isalnum())
        return content_chars / len(text)

    def check_coherence(self, text: str) -> float:
        """Check if text appears to be coherent language."""
        words = text.split()
        if len(words) < 10:
            return 0.5

        # Check average word length (garbled text often has weird lengths)
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 15:
            return 0.5

        # Check for reasonable sentence patterns
        sentences = text.split(".")
        if len(sentences) < 2:
            return 0.7

        return 1.0
```

### 4.5 Complete Pipeline

```python
class DataCleanupPipeline:
    """Complete data cleanup pipeline for RAG."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.parser = UnifiedDocumentParser(
            ocr_enabled=self.config.get("ocr_enabled", False)
        )
        self.cleaner = TextCleaner()
        self.noise_remover = NoiseRemover()
        self.enricher = MetadataEnricher()
        self.validator = QualityValidator(
            min_score=self.config.get("min_quality_score", 0.6)
        )

    def process(self, file_path: str) -> Optional[dict]:
        """Process a document through the complete pipeline."""

        # Step 1: Parse
        try:
            parsed = self.parser.parse(file_path)
        except Exception as e:
            return {"error": f"Parse failed: {e}", "file": file_path}

        # Step 2: Clean
        cleaned_text = self.cleaner.clean(parsed.content)

        # Step 3: Remove noise
        denoised_text = self.noise_remover.clean(cleaned_text)

        # Step 4: Validate quality
        quality_report = self.validator.validate(denoised_text)

        if not quality_report.passed:
            return {
                "error": "Quality check failed",
                "file": file_path,
                "issues": quality_report.issues,
                "suggestions": quality_report.suggestions
            }

        # Step 5: Enrich metadata
        metadata = self.enricher.enrich(denoised_text, {
            "source": file_path,
            "filename": Path(file_path).name,
            "file_type": Path(file_path).suffix
        })

        return {
            "content": denoised_text,
            "metadata": metadata,
            "tables": parsed.tables,
            "quality_score": quality_report.score
        }

    def process_batch(self, file_paths: list, parallel: bool = True) -> list:
        """Process multiple documents."""
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(self.process, file_paths))
        else:
            results = [self.process(fp) for fp in file_paths]

        return results
```

[↑ Back to Top](#table-of-contents)

---

## 5. Problem Statements & Solutions

### Problem 1: Scanned PDFs Return Empty Text

**Symptoms:**
- PDF parsing returns empty or minimal text
- Content visible in PDF viewer but not extracted

**Root Cause:** PDF contains scanned images, not text layer

**Solution:**

```python
import pytesseract
from pdf2image import convert_from_path

def extract_scanned_pdf(file_path: str, dpi: int = 300) -> str:
    """Extract text from scanned PDF using OCR."""

    # Convert PDF pages to images
    images = convert_from_path(file_path, dpi=dpi)

    text_parts = []
    for i, image in enumerate(images):
        # Preprocess image for better OCR
        image = preprocess_for_ocr(image)

        # Run OCR
        text = pytesseract.image_to_string(
            image,
            config='--psm 1 --oem 3'  # Automatic page segmentation, best OCR engine
        )
        text_parts.append(f"[Page {i+1}]\n{text}")

    return "\n\n".join(text_parts)

def preprocess_for_ocr(image):
    """Preprocess image to improve OCR accuracy."""
    import cv2
    import numpy as np

    # Convert to numpy array
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)

    return Image.fromarray(denoised)
```

**Cost-Effective Alternative:** Use Unstructured.io with `strategy="hi_res"` which includes OCR.

---

### Problem 2: Tables Extracted as Garbled Text

**Symptoms:**
- Table data is jumbled
- Columns merged together
- Row alignment lost

**Root Cause:** Standard text extraction doesn't preserve table structure

**Solution:**

```python
import pdfplumber
import pandas as pd

def extract_tables_properly(file_path: str) -> list[pd.DataFrame]:
    """Extract tables with structure preserved."""

    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Find tables on page
            page_tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            })

            for table in page_tables:
                if table:
                    # Convert to DataFrame
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)

    return tables

def table_to_text(df: pd.DataFrame) -> str:
    """Convert table to readable text format for embedding."""

    # Option 1: Markdown format
    markdown = df.to_markdown(index=False)

    # Option 2: Row-by-row description
    descriptions = []
    columns = df.columns.tolist()

    for _, row in df.iterrows():
        desc = "; ".join([f"{col}: {row[col]}" for col in columns])
        descriptions.append(desc)

    return "\n".join(descriptions)
```

---

### Problem 3: HTML Full of Boilerplate

**Symptoms:**
- Extracted text includes navigation, ads, footers
- Actual content buried in noise
- Poor retrieval quality

**Solution:**

```python
from trafilatura import extract
from readability import Document
import requests

def extract_main_content(url_or_html: str) -> str:
    """Extract only main content from HTML."""

    # Get HTML
    if url_or_html.startswith("http"):
        response = requests.get(url_or_html)
        html = response.text
    else:
        html = url_or_html

    # Method 1: Trafilatura (best for articles)
    content = extract(
        html,
        include_tables=True,
        include_comments=False,
        include_images=False,
        no_fallback=False
    )

    if content and len(content) > 100:
        return content

    # Method 2: Readability fallback
    doc = Document(html)
    return doc.summary()

def clean_extracted_html(text: str) -> str:
    """Additional cleaning for HTML content."""

    # Remove common artifacts
    patterns_to_remove = [
        r"Cookie\s*(Policy|Notice|Consent).*?\n",
        r"Subscribe\s+to\s+.*?\n",
        r"Share\s+(on|via)\s+.*?\n",
        r"Follow\s+us\s+on.*?\n",
        r"Advertisement\s*\n",
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text
```

---

### Problem 4: Encoding Issues (Mojibake)

**Symptoms:**
- Strange characters: Ã©, â€™, Â
- Unreadable text sections
- Mixed encoding artifacts

**Solution:**

```python
import ftfy
import chardet

def fix_encoding_issues(text: str) -> str:
    """Fix common encoding problems."""

    # Use ftfy for automatic fixing
    fixed = ftfy.fix_text(text)

    return fixed

def detect_and_decode(file_path: str) -> str:
    """Detect encoding and decode properly."""

    # Read raw bytes
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    # Detect encoding
    detection = chardet.detect(raw_bytes)
    encoding = detection["encoding"]
    confidence = detection["confidence"]

    print(f"Detected encoding: {encoding} (confidence: {confidence})")

    # Decode with detected encoding
    if confidence > 0.7:
        text = raw_bytes.decode(encoding, errors="replace")
    else:
        # Try common encodings
        for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except:
                continue

    # Apply ftfy fix
    return ftfy.fix_text(text)
```

---

### Problem 5: Duplicate Content

**Symptoms:**
- Same document indexed multiple times
- Near-duplicate paragraphs from different versions
- Redundant retrieval results

**Solution:**

```python
import hashlib
from datasketch import MinHash, MinHashLSH

class DeduplicationManager:
    """Detect and handle duplicate content."""

    def __init__(self, similarity_threshold: float = 0.8):
        self.threshold = similarity_threshold
        self.lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
        self.seen_hashes = set()

    def get_exact_hash(self, text: str) -> str:
        """Get exact content hash."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get_minhash(self, text: str) -> MinHash:
        """Get MinHash for similarity detection."""
        mh = MinHash(num_perm=128)

        # Shingle the text (3-word chunks)
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i:i+3])
            mh.update(shingle.encode())

        return mh

    def is_duplicate(self, text: str, doc_id: str) -> tuple[bool, list]:
        """Check if document is duplicate."""

        # Exact duplicate check
        exact_hash = self.get_exact_hash(text)
        if exact_hash in self.seen_hashes:
            return True, ["exact_duplicate"]

        # Near-duplicate check
        mh = self.get_minhash(text)
        similar_docs = self.lsh.query(mh)

        if similar_docs:
            return True, list(similar_docs)

        # Not a duplicate - add to index
        self.seen_hashes.add(exact_hash)
        self.lsh.insert(doc_id, mh)

        return False, []

    def deduplicate_batch(self, documents: list[dict]) -> list[dict]:
        """Remove duplicates from a batch of documents."""
        unique_docs = []

        for doc in documents:
            is_dup, similar = self.is_duplicate(doc["content"], doc["id"])

            if not is_dup:
                unique_docs.append(doc)
            else:
                print(f"Skipping duplicate: {doc['id']} (similar to: {similar})")

        return unique_docs
```

[↑ Back to Top](#table-of-contents)

---

## 6. Trade-offs

### Parsing Strategy Trade-offs

| Strategy | Speed | Quality | Cost | Best For |
|----------|-------|---------|------|----------|
| **Simple extraction** | ⚡⚡⚡ | ⭐⭐ | Free | Plain text, simple PDFs |
| **Structure-aware** | ⚡⚡ | ⭐⭐⭐ | Free | Documents with sections |
| **OCR (local)** | ⚡ | ⭐⭐⭐ | Free | Scanned documents |
| **OCR (cloud)** | ⚡⚡ | ⭐⭐⭐⭐ | $$$ | High accuracy needed |
| **Vision models** | ⚡ | ⭐⭐⭐⭐⭐ | $$$$ | Complex layouts |

### Cleaning Aggressiveness Trade-offs

| Level | What's Removed | Risk | Best For |
|-------|----------------|------|----------|
| **Minimal** | Only obvious noise | May retain noise | Precise domains |
| **Moderate** | Common patterns | Balanced | General use |
| **Aggressive** | All non-content | May lose info | Noisy sources |

### Quality vs Speed Trade-offs

```
High Quality                              High Speed
────────────────────────────────────────────────────
│                                                   │
│  Vision AI    OCR    Structure    Simple    Regex │
│  Parsing      Full   Aware        Extract   Only  │
│                                                   │
│  Slowest ◄─────────────────────────────► Fastest │
│  Highest ◄─────────────────────────────► Lowest  │
│  Quality                                  Quality │
│                                                   │
────────────────────────────────────────────────────
```

[↑ Back to Top](#table-of-contents)

---

## 7. Cost-Effective Solutions

### Free & Open Source Stack

| Component | Free Option | Notes |
|-----------|-------------|-------|
| **PDF Parsing** | pdfplumber, PyMuPDF | Good for most PDFs |
| **OCR** | Tesseract + preprocessing | Quality depends on image |
| **HTML** | Trafilatura, BeautifulSoup | Excellent quality |
| **Word/Excel** | python-docx, pandas | Native format support |
| **All-in-one** | Unstructured.io (local) | Best free option |

### When to Pay

| Scenario | Free Works | Consider Paid |
|----------|------------|---------------|
| Simple documents | ✅ | - |
| Scanned PDFs | ⚠️ | Azure/AWS OCR |
| Complex tables | ⚠️ | Document AI |
| High volume | ⚠️ | Managed service |
| Legal/Medical | ❌ | Specialized parsers |

### Cost Comparison

```
Monthly cost for processing 10,000 documents:

Free Stack (Self-hosted):
├── Unstructured.io local    $0
├── Tesseract OCR            $0
├── Server (4 CPU, 16GB)     ~$50/month
└── Total                    ~$50/month

Cloud OCR:
├── Azure Form Recognizer    ~$150/month
├── AWS Textract             ~$150/month
└── Google Document AI       ~$150/month

Hybrid (Recommended):
├── Simple docs: Local       $0
├── Complex docs: Cloud      ~$50/month (5K docs)
└── Total                    ~$100/month
```

### Optimization Tips

1. **Route by Complexity**
   ```python
   def route_parser(file_path: str) -> str:
       # Check if PDF has text layer
       if is_pdf_with_text(file_path):
           return "local_simple"
       elif is_scanned_pdf(file_path):
           if is_high_quality_scan(file_path):
               return "local_ocr"
           else:
               return "cloud_ocr"
       else:
           return "local_simple"
   ```

2. **Batch Processing** - Send multiple pages in one API call

3. **Cache Results** - Don't re-parse unchanged documents

4. **Prefilter** - Skip documents that don't need processing

[↑ Back to Top](#table-of-contents)

---

## 8. Best Practices

### DO's

1. **Always Validate Output**
   ```python
   result = parse(document)
   if not validate_quality(result):
       log_warning(f"Low quality extraction: {document}")
       try_alternative_parser(document)
   ```

2. **Preserve Structure**
   - Keep section headers
   - Maintain list formatting
   - Preserve table structure

3. **Add Rich Metadata**
   ```python
   metadata = {
       "source": "policy_docs",
       "document_type": "policy",
       "version": "2.1",
       "effective_date": "2024-01-01",
       "department": "legal",
       "confidentiality": "internal"
   }
   ```

4. **Handle Failures Gracefully**
   ```python
   try:
       result = parse(document)
   except Exception as e:
       log_error(f"Parse failed: {e}")
       result = fallback_parse(document)
   ```

5. **Version Your Pipeline**
   - Track processing configuration
   - Enable reprocessing when pipeline improves

### DON'Ts

1. **Don't Over-Clean**
   - Keep technical terms
   - Preserve meaningful punctuation
   - Don't remove all special characters

2. **Don't Ignore Errors**
   - Log all parsing failures
   - Track quality metrics
   - Alert on quality drops

3. **Don't Assume Format**
   - Validate file type
   - Handle edge cases
   - Test with real data

4. **Don't Skip Testing**
   - Create test documents
   - Validate extraction quality
   - Compare parser outputs

[↑ Back to Top](#table-of-contents)

---

## 9. Quick Reference

### Parser Selection

```
Document Type → Best Parser
───────────────────────────────────────
Simple PDF      → pdfplumber
Complex PDF     → Unstructured.io (hi_res)
Scanned PDF     → Tesseract or Cloud OCR
Word (.docx)    → python-docx
Excel           → pandas
HTML            → Trafilatura
Markdown        → markdown-it
All Types       → Unstructured.io
```

### Cleaning Checklist

```
□ Fix encoding (ftfy)
□ Normalize whitespace
□ Remove headers/footers
□ Handle special characters
□ Deduplicate content
□ Validate quality
□ Add metadata
```

### Quality Thresholds

```
Metric              Minimum    Target
───────────────────────────────────────
Content length      100 chars  500+ chars
Quality score       0.6        0.8+
Encoding issues     <5%        0%
Content density     >0.3       >0.5
```

### File Extension Mapping

```python
PARSER_MAP = {
    # Text
    ".txt": "text",
    ".md": "markdown",

    # Documents
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc_legacy",
    ".rtf": "rtf",

    # Spreadsheets
    ".xlsx": "excel",
    ".xls": "excel_legacy",
    ".csv": "csv",

    # Presentations
    ".pptx": "pptx",
    ".ppt": "ppt_legacy",

    # Web
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",

    # Email
    ".eml": "email",
    ".msg": "outlook",
}
```

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| - | [Main Guide](../README.md) | [Chunking Strategies →](./02-chunking.md) |

---

*Last updated: 2024*
