# GLiNER2 Long-Context Extraction Tutorial

Learn how to extract information from documents that are longer than the model's normal context window.

## Table of Contents
- [Why Use Long-Context Extraction](#why-use-long-context-extraction)
- [Entity Extraction from Long Documents](#entity-extraction-from-long-documents)
- [Choosing Chunk Settings](#choosing-chunk-settings)
- [Generic Schema Extraction](#generic-schema-extraction)
- [Batch Long-Document Extraction](#batch-long-document-extraction)
- [Best Practices](#best-practices)

## Why Use Long-Context Extraction

The standard `extract(...)` and `extract_entities(...)` APIs process one input at a time. If you pass `max_len`, GLiNER2 truncates the text to the first N word tokens. That is useful for speed, but it means entities near the middle or end of a long document can be missed.

Long-context extraction is explicit opt-in. It:
- Splits the document into overlapping word chunks.
- Runs normal GLiNER2 inference on each chunk.
- Remaps chunk-local spans back to the original document.
- Merges duplicate detections from overlapping chunks.

Use it for reports, contracts, support tickets, transcripts, PDFs converted to text, logs, and other multi-page documents.

## Entity Extraction from Long Documents

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

long_text = """
Annual report background and financial commentary...

Apple CEO Tim Cook announced Vision Pro updates in Cupertino
during September 2025.

More report text continues for many pages...
"""

result = extractor.extract_entities_long(
    long_text,
    ["company", "person", "product", "location", "date"],
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
    include_confidence=True,
)

print(result)
# {
#     "entities": {
#         "company": [{"text": "Apple", "confidence": 0.99, "start": 52, "end": 57}],
#         "person": [{"text": "Tim Cook", "confidence": 0.99, "start": 62, "end": 70}],
#         "product": [{"text": "Vision Pro", "confidence": 0.99, "start": 81, "end": 91}],
#         "location": [{"text": "Cupertino", "confidence": 0.99, "start": 103, "end": 112}],
#         "date": [{"text": "September 2025", "confidence": 0.99, "start": 120, "end": 134}]
#     }
# }
```

When `include_spans=True`, `start` and `end` are global character offsets into the original `long_text`, not offsets inside a chunk.

```python
for label, entities in result["entities"].items():
    for entity in entities:
        assert long_text[entity["start"]:entity["end"]] == entity["text"]
```

## Choosing Chunk Settings

Two parameters control how the document is scanned:

- `chunk_size`: Maximum number of word tokens per chunk.
- `chunk_overlap`: Number of word tokens repeated between adjacent chunks.

Good starting values:

```python
result = extractor.extract_entities_long(
    long_text,
    ["company", "person", "product"],
    chunk_size=384,
    chunk_overlap=64,
)
```

Increase `chunk_overlap` when important mentions may appear near chunk boundaries, especially for multi-token entities or relations. Keep `chunk_overlap` smaller than `chunk_size`.

For faster inference, reduce `chunk_size` or `chunk_overlap`. For better recall, increase overlap and use precise label descriptions.

```python
entity_types = {
    "contract_party": "Names of companies or people that are parties to the agreement",
    "effective_date": "Dates when the agreement starts or becomes valid",
    "termination_clause": "Text describing when or how the agreement can end",
}

result = extractor.extract_entities_long(
    contract_text,
    entity_types,
    chunk_size=512,
    chunk_overlap=96,
    include_spans=True,
)
```

## Generic Schema Extraction

Use `extract_long(...)` with any GLiNER2 schema.

```python
schema = (
    extractor.create_schema()
    .entities({
        "company": "Company names",
        "person": "Names of people",
        "product": "Product names",
    })
    .classification("document_type", ["report", "contract", "email", "support_ticket"])
)

result = extractor.extract_long(
    long_text,
    schema,
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
    include_confidence=True,
)
```

Span-based outputs are merged across chunks. Classification outputs are aggregated across chunk predictions, preferring higher confidence when scores are available.

## Batch Long-Document Extraction

Use `batch_extract_entities_long(...)` for multiple long documents with the same entity schema.

```python
documents = [
    open("report_2024.txt").read(),
    open("report_2025.txt").read(),
]

results = extractor.batch_extract_entities_long(
    documents,
    ["company", "person", "product", "location", "date"],
    batch_size=8,
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
)

for doc, result in zip(documents, results):
    for entities in result["entities"].values():
        for entity in entities:
            assert doc[entity["start"]:entity["end"]] == entity["text"]
```

Use `batch_extract_long(...)` when each document has a full schema, or when each document needs a different schema.

```python
schemas = [
    extractor.create_schema().entities(["company", "date"]),
    extractor.create_schema().entities(["person", "location"]),
]

results = extractor.batch_extract_long(
    documents,
    schemas,
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
)
```

## Best Practices

- Prefer `extract_entities_long(...)` for long entity extraction instead of `extract_entities(..., max_len=...)`; `max_len` truncates, while long-context extraction scans the full document.
- Use descriptive labels to reduce noisy matches in repetitive documents.
- Request spans during development so you can verify that offsets point to the intended text.
- Tune `threshold`, `chunk_size`, and `chunk_overlap` on a small validation set from your domain.
- For very repetitive text, expect repeated valid mentions. GLiNER2 deduplicates overlap artifacts, but keeps distinct mentions at different document positions.
- Keep long-context extraction explicit in production pipelines so latency and recall trade-offs are easy to reason about.
