# /ingest API Guide

This document explains how to use the `/ingest` API, the accepted JSON shapes, options, responses, and provides examples.

---

## Accepted JSON Shapes

You can send data in three shapes (JSON), with optional chunking and ID control:

1. **Documents + Metadatas**
   - **documents**: list of strings
   - **metadatas**: single dict (broadcast) or list of dicts (1:1)

2. **Items**
   - **items**: list of `{ text, metadata }` objects

---

## Extra Options

- **chunk**: `bool` (default `true`)
- **chunk_size**: default `1200`
- **overlap**: default `200`
- **id_fields**: e.g., `["path","chunk"]` — included in the ID so identical text from different files stays distinct

---

## Responses

Example response:
```json
{
  "received_docs": 5,
  "added_docs": 5,
  "skipped_duplicates": 0
}
```

---

## cURL Examples (bash)

### 1. Documents + One Metadata (broadcast to all)
```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{
    "documents": [
      "Deploy notes: use docker compose.",
      "CI tips: cache your layers."
    ],
    "metadatas": {
      "source": "api",
      "path": "/notes/eng/guide.md",
      "filename": "guide.md",
      "tags": "devops,notes"
    },
    "id_fields": ["path","chunk"]
  }'
```

### 2. Documents + Per-doc Metadatas (1:1)
```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{
    "documents": [
      "Alpha file: part 1",
      "Beta file: part 1"
    ],
    "metadatas": [
      { "source":"api", "path":"/notes/alpha.md", "filename":"alpha.md" },
      { "source":"api", "path":"/notes/beta.md",  "filename":"beta.md"  }
    ],
    "id_fields": ["path","chunk"]
  }'
```

### 3. Items (each with its own metadata)
```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{
    "items": [
      { "text":"Discord bot commands for Luna...", "metadata":{"source":"api","path":"/kb/bot.md","filename":"bot.md","topic":"discord"} },
      { "text":"RAG setup notes...", "metadata":{"source":"api","path":"/kb/rag.md","filename":"rag.md","topic":"rag"} }
    ],
    "id_fields": ["path","chunk"]
  }'
```

### 4. Turn Chunking Off (store as single docs)
```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{
    "documents": ["This is stored as one whole doc."],
    "metadatas": { "source":"api", "path":"/notes/whole.txt", "filename":"whole.txt" },
    "chunk": false,
    "id_fields": ["path"]
  }'
```

### 5. Tiny Chunks to See Splitting Clearly
```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{
    "items": [
      { "text":"Line A. Line B. Line C. Line D. Line E.", "metadata":{"source":"api","path":"/toy/chunks.txt","filename":"chunks.txt"} }
    ],
    "chunk": true,
    "chunk_size": 12,
    "overlap": 4,
    "id_fields": ["path","chunk"]
  }'
```

---

## PowerShell (Windows) Example

Use double quotes for JSON and escape inner quotes with backticks:

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest `
  -H "Content-Type: application/json" `
  --data-raw "{
    ""documents"": [""Hello from PS1"", ""Another doc""],
    ""metadatas"": { ""source"": ""api"", ""path"": ""/ps/notes.md"", ""filename"": ""notes.md"" },
    ""id_fields"": [""path"", ""chunk""]
  }"
```

---

## Tips

- Good metadata fields: `source`, `path`, `filename`, `chunk`, `tags`, `created_at`.
- If you see `skipped_duplicates > 0`, that’s dedupe working (same text+id_fields combo).
- Use `/docs` (Swagger UI) to try the endpoint interactively.
- If retriever returns nothing, loosen filters (e.g., disable distance cutoff) and verify `collection.count() > 0`.

---
