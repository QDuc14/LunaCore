import hashlib

import chromadb

from app.services.llm_provider import embed_texts

from pathlib import Path

from functools import lru_cache

from typing import List, Dict, Optional, Union, Any

from chromadb import PersistentClient

# --- Client & collection setup ---
CHROMA_PATH = str(Path("chromadb_store").resolve())

try:
    chroma_client = PersistentClient(path=CHROMA_PATH)  # preferred on >=0.5
except Exception:
    # Fallback for older versions
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))

collection = chroma_client.get_or_create_collection(
    name="luna-memory",
    metadata={"hnsw:space": "cosine"},
    embedding_function=None
)


# --- Embedding helper ---
def get_embedding_for_text(text: str) -> list[float]:
    return asyncio.run(embed_texts([text]))[0]

# --- Cached query embed (used by retriever) ---
@lru_cache(maxsize=2048)
def _cached_query_embedding(text: str) -> tuple:
    # Compute via provider; cache as tuple
    import asyncio
    emb = asyncio.run(embed_texts([text]))[0]
    return tuple(float(x) for x in emb)

def get_query_embedding(text: str) -> list[float]:
    """Public for retriever: cached, returns a list."""
    return list(_cached_query_embedding(text))

def clear_embedding_cache() -> None:
    _cached_query_embedding.cache_clear()

# --- Upsert function ---
def _batch(iterable: List[Any], size: int) -> List[List[Any]]:
    return [iterable[i:i + size] for i in range(0, len(iterable), size)]

def _make_id(doc: str, meta: Dict[str, Any], id_fields: Optional[List[str]]) -> str:
    """
    Build a stable ID:
    - By default, hash(doc)
    - If id_fields provided, hash(doc + selected meta values) so the same text
      from different files/chunks can be distinct.
    """
    if id_fields:
        key_parts = [doc] + [str(meta.get(f, "")) for f in id_fields]
        key = "||".join(key_parts)
    else:
        key = doc
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def embed_and_store(
    documents: List[str],
    metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    *,
    id_fields: Optional[List[str]] = None,
    existence_check_batch: int = 512,
    embed_batch: int = 128,
) -> int:
    """
    Add only-new docs to Chroma, returning the count added.

    Args:
      documents: list of raw text chunks.
      metadatas: EITHER a single dict (broadcast) OR a list[dict] aligned to documents.
      id_fields: optional list of metadata keys to include in the content hash.
                 Use this to treat identical text from different sources as distinct,
                 e.g., id_fields=['path','chunk'].
      existence_check_batch: batch size for .get(ids=...).
      embed_batch: batch size for embedding + add.

    Notes:
      - Requires the global 'collection' and 'get_embedding_for_text' to be defined.
      - Collection should be created with embedding_function=None.
    """
    if not documents:
        return 0

    # Normalize metadatas → list[dict]
    if metadatas is None:
        meta_list: List[Dict[str, Any]] = [{"source": "ingest"} for _ in documents]
    elif isinstance(metadatas, dict):
        meta_list = [metadatas for _ in documents]
    else:
        meta_list = metadatas
        if len(meta_list) != len(documents):
            raise ValueError("Length of 'metadatas' must match 'documents' when a list is provided.")

    # 1) In-batch dedupe by computed ID
    id_to_doc_meta: Dict[str, Dict[str, Any]] = {}
    for doc, meta in zip(documents, meta_list):
        h = _make_id(doc, meta, id_fields)
        if h not in id_to_doc_meta:
            id_to_doc_meta[h] = {"doc": doc, "meta": meta}

    candidate_ids = list(id_to_doc_meta.keys())

    # 2) Batch-check which IDs already exist
    existing_ids = set()
    for chunk in _batch(candidate_ids, existence_check_batch):
        try:
            got = collection.get(ids=chunk)
            for eid in got.get("ids", []) or []:
                existing_ids.add(eid)
        except Exception:
            # Fail-open if old client behaves differently
            pass

    to_add_ids = [i for i in candidate_ids if i not in existing_ids]
    if not to_add_ids:
        return 0

    to_add_docs = [id_to_doc_meta[i]["doc"] for i in to_add_ids]
    to_add_metas = [id_to_doc_meta[i]["meta"] for i in to_add_ids]

    # 3) Embed & add in batches (keeping per-doc metadatas)
    added = 0
    for id_chunk, doc_chunk, meta_chunk in zip(
        _batch(to_add_ids, embed_batch),
        _batch(to_add_docs, embed_batch),
        _batch(to_add_metas, embed_batch),
    ):
        embeddings = [get_embedding_for_text(d) for d in doc_chunk]
        collection.add(
            ids=id_chunk,
            documents=doc_chunk,
            metadatas=meta_chunk,
            embeddings=embeddings,
        )
        added += len(id_chunk)

    # 4) Persist (noop on PersistentClient, safe elsewhere)
    try:
        chroma_client.persist()  # type: ignore[attr-defined]
    except Exception:
        pass

    return added
