from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import hashlib

from app.rag.embedder import embed_and_store, collection
from app.rag.embedder import get_query_embedding  # for retrieve_memories


def _now_utc() -> datetime: return datetime.now(timezone.utc)
def _iso(dt: datetime) -> str: return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def _scope_key(user_id: Optional[str], conversation_id: str) -> str:
    return (user_id or "").strip() or f"conv:{conversation_id}"

def bullets_from_summary(summary: str) -> List[str]:
    out: List[str] = []
    for raw in (summary or "").splitlines():
        line = raw.strip("•-• \t\r\n").strip()
        if line:
            out.append((line[:240] + "…") if len(line) > 240 else line)
    return out[:8]

def _get_existing_by_key(memory_key: str) -> Dict[str, Any] | None:
    # use where to fetch metadata; don't rely on returning ids from get()
    try:
        res = collection.get(
            where={"$and": [{"topic": {"$eq": "memory"}}, {"memory_key": {"$eq": memory_key}}]},
            include=["metadatas"],
            limit=1,
        )
    except Exception:
        return None
    metas = res.get("metadatas") or []
    metas = metas[0] if metas and isinstance(metas[0], list) else metas
    if not metas:
        return None
    # a follow-up get to retrieve IDs by content filter is not portable; we update by re-add with same id_fields
    return {"id": None, "meta": metas[0] or {}}

def save_memories_from_summary(
    summary: str,
    conversation_id: str,
    user_id: Optional[str] = None,
    ttl_days: int = 30,
    source: str = "auto-memory",
    version: str = "v1",
) -> int:
    bullets = bullets_from_summary(summary)
    if not bullets:
        return 0
    now = _now_utc()
    expires_at = _iso(now + timedelta(days=max(1, ttl_days)))
    scope = _scope_key(user_id, conversation_id)

    new_docs: List[str] = []
    new_metas: List[Dict[str, Any]] = []

    for b in bullets:
        key = hashlib.sha256(f"{scope}|{b}".encode("utf-8")).hexdigest()
        existing = _get_existing_by_key(key)
        if existing:
            # refresh via re-add with same memory_key (embed_and_store dedups by id_fields)
            new_docs.append(b)
            new_metas.append({
                "topic":"memory","user_id":user_id,"conversation_id":conversation_id,"scope":scope,
                "memory_key":key,"source":source,"version":version,"created_at":_iso(now),
                "expires_at":expires_at,"ttl_days":ttl_days,"seen_count":int((existing["meta"] or {}).get("seen_count", 0)) + 1,
            })
        else:
            new_docs.append(b)
            new_metas.append({
                "topic":"memory","user_id":user_id,"conversation_id":conversation_id,"scope":scope,
                "memory_key":key,"source":source,"version":version,"created_at":_iso(now),
                "expires_at":expires_at,"ttl_days":ttl_days,"seen_count":1,
            })

    if not new_docs:
        return 0
    added = embed_and_store(new_docs, metadatas=new_metas, id_fields=["memory_key"])
    return added

def retrieve_memories(
    query: str,
    conversation_id: str,
    user_id: Optional[str] = None,
    k: int = 4,
    refresh_on_recall: bool = True,
    ttl_days: int = 30,
) -> List[str]:
    q_emb = get_query_embedding(query)
    scope = _scope_key(user_id, conversation_id)
    where = {"$and":[
        {"topic":{"$eq":"memory"}},
        {"$or":[{"scope":{"$eq":scope}}, {"conversation_id":{"$eq":conversation_id}}]}
    ]}
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents","metadatas","distances"],
        where=where,
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0] if res.get("metadatas") else [{} for _ in docs]

    now_iso = _iso(_now_utc())
    new_exp = _iso(_now_utc() + timedelta(days=max(1, ttl_days)))

    out_docs: List[str] = []
    for d, m in zip(docs or [], metas or []):
        if not d:
            continue
        exp = (m or {}).get("expires_at", now_iso)
        if exp < now_iso:
            continue
        out_docs.append(d)

        if refresh_on_recall:
            # refresh by re-add with same memory_key → embed_and_store dedups and updates metadata
            key = (m or {}).get("memory_key")
            if key:
                embed_and_store([d], metadatas=[{
                    **(m or {}),
                    "expires_at": new_exp,
                    "last_recalled_at": now_iso,
                    "seen_count": int((m or {}).get("seen_count", 0)) + 1,
                }], id_fields=["memory_key"])

    return out_docs
