import ollama
import json

from pathlib import Path
from typing import List, Dict, Iterable, Tuple
from app.rag.embedder import collection, embed_and_store, get_embedding_for_text, get_query_embedding

# --- Simple chunker (char-based with overlap) ---
def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# --- Ingest one file ---
TEXT_EXTS = {".txt", ".md", ".mdx", ".py", ".js", ".ts", ".tsx", ".json", ".html", ".css"}

def ingest_file(path: Path, source_tag: str = "fs") -> int:
    if not path.exists() or path.suffix.lower() not in TEXT_EXTS:
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    chunks = split_text(text)
    docs = []
    metas = []
    for i, ch in enumerate(chunks):
        docs.append(ch)
        metas.append({
            "source": source_tag,
            "path": str(path),
            "chunk": i,
            "filename": path.name
        })
    # embed_and_store uses a single metadata per call; call per chunk batch
    # to preserve per-chunk metadata we add in small groups
    added = 0
    batch_docs: List[str] = []
    batch_meta: Dict[str, str] = {}
    for d, m in zip(docs, metas):
        batch_docs.append(d)
        # Pass one metadata dict; embedder applies the same meta to each doc,
        # so we encode per-chunk meta inside the text header to preserve fields.
        # Alternative: modify embedder to accept 'metadatas' list.
    # Better: slight tweak—call collection.add directly so we keep metadatas:
    embs = [get_query_embedding(d) for d in docs]
    from hashlib import sha256
    ids = [sha256(d.encode("utf-8")).hexdigest() for d in docs]
    # Filter out existing ids
    got = collection.get(ids=ids)
    existing = set(got.get("ids", []))
    to_add = [(i, d, m, e) for i, d, m, e in zip(ids, docs, metas, embs) if i not in existing]
    if to_add:
        collection.add(
            ids=[t[0] for t in to_add],
            documents=[t[1] for t in to_add],
            metadatas=[t[2] for t in to_add],
            embeddings=[t[3] for t in to_add],
        )
        added = len(to_add)
    try:
        # Persist if available (noop on PersistentClient)
        collection._client.persist()  # type: ignore[attr-defined]
    except Exception:
        pass
    return added

def ingest_directory(dir_path: str, source_tag: str = "fs") -> int:
    p = Path(dir_path)
    count = 0
    for ext in TEXT_EXTS:
        for f in p.rglob(f"*{ext}"):
            count += ingest_file(f, source_tag=source_tag)
    return count

# --- Query memory ---
def query_memory(query: str, k: int = 5):
    q_emb = get_query_embedding(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["ids", "documents", "metadatas", "distances"],
    )
    # Normalize output
    hits = []
    if res and res.get("ids"):
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
    return hits

# --- RAG answer with Ollama ---
def ask_luna(question: str, k: int = 5, model: str = "luna") -> str:
    hits = query_memory(question, k=k)
    context = "\n\n".join(
        f"[{h['metadata'].get('filename','memo')}#{h['metadata'].get('chunk',0)}]\n{h['document']}"
        for h in hits
    )
    system = "You are Luna, a concise helpful assistant. Use the provided context if relevant; if not, say you don't find it."
    prompt = (
        f"Context:\n{context}\n\n"
        f"User question: {question}\n\n"
        "Instructions: Cite filenames and chunk numbers from the context when you use them."
    )
    resp = ollama.chat(model=model, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ])
    return resp["message"]["content"]

# --- Utilities ---
def backup_store(out_path: str = "chroma_backup.json") -> int:
    """Dump all docs & metas for backup/debug (small to medium collections)."""
    # Warning: for very large stores, this may be slow.
    try:
        count = collection.count()
        # Peek returns only a few; for full export we need to page via where={}.
        # chroma 0.5+ supports get with limit/offset
        offset = 0
        page = 200
        all_items = []
        while True:
            got = collection.get(limit=page, offset=offset, include=["documents","metadatas","ids"])
            ids = got.get("ids", [])
            if not ids:
                break
            for i in range(len(ids)):
                all_items.append({
                    "id": ids[i],
                    "document": got["documents"][i],
                    "metadata": got["metadatas"][i],
                })
            offset += page
        Path(out_path).write_text(json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(all_items)
    except Exception:
        return 0

if __name__ == "__main__":
    # Example: ingest your notes folder, then ask
    print("Ingested:", ingest_directory("./notes"))
    print(ask_luna("What did I say about Luna's tone?"))