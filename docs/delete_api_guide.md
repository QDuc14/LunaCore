# /delete API Guide

This document explains how to use the `/delete` API to remove previously ingested documents. It is aligned with the FastAPI implementation provided.

---

## Accepted JSON Shapes

You can delete documents using one of the following request bodies:

1. **By IDs**
   ```json
   {
     "ids": ["doc_123", "doc_456"]
   }
   ```
   Deletes documents explicitly by ID.

2. **By Metadata Filter**
   ```json
   {
     "where": { "path": "/persona/core.md" }
   }
   ```
   Deletes documents matching metadata.  
   - Operators supported: equality and special operators like `{"topic": {"$eq":"persona"}}`.

3. **By Document Content Filter**
   ```json
   {
     "where_document": { "$contains": "cosmic spark" }
   }
   ```
   Deletes documents that contain a text match.

---

## Extra Options

- **dry_run**: `bool`  
  If `true`, no documents are deleted, but you get a count of what *would* be deleted.

- **page_size**: `int`  
  Controls how many IDs are fetched per batch when resolving filters.

---

## Responses

The API returns a `DeleteResponse`:

```json
{
  "method": "where_document",
  "matched": 12,
  "deleted": 12,
  "details": {
    "note": "dry_run"
  }
}
```

- **method**: which strategy was used (`ids`, `where`, `where_document`)
- **matched**: number of documents matching the request
- **deleted**: number of documents actually deleted
- **details**: optional metadata (e.g., `{ "note": "dry_run" }`)

---

## cURL Examples (bash)

### 1) Delete by IDs
```bash
curl -X POST http://127.0.0.1:8000/delete   -H "Content-Type: application/json"   -d '{
    "ids": ["HGN2508000905:0", "HGN2508000905:1"]
  }'
```

### 2) Delete by metadata filter
```bash
curl -X POST http://127.0.0.1:8000/delete   -H "Content-Type: application/json"   -d '{
    "where": { "topic": {"$eq":"persona"} }
  }'
```

### 3) Delete by document content
```bash
curl -X POST http://127.0.0.1:8000/delete   -H "Content-Type: application/json"   -d '{
    "where_document": { "$contains": "cosmic spark" }
  }'
```

### 4) Dry run (preview only)
```bash
curl -X POST http://127.0.0.1:8000/delete   -H "Content-Type: application/json"   -d '{
    "where": { "path": "/persona/core.md" },
    "dry_run": true
  }'
```

---

## Dangerous: Drop Entire Collection

You can delete an entire collection with:

```bash
curl -X POST http://127.0.0.1:8000/collections/drop   -H "Content-Type: application/json"   -d '{"name": "memory"}'
```

Response:
```json
{ "status": "Collection 'luna-memory' deleted." }
```

⚠️ This permanently deletes all documents in the collection.

---

## Tips

- Start with `dry_run` to confirm the scope before deleting.  
- Prefer `where` or `where_document` filters when removing many related chunks.  
- If your retriever still returns content after deletion, verify you deleted from the correct collection/namespace.  

---
