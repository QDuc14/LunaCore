from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from app.rag.embedder import collection, get_query_embedding


# ---------- Types ----------

class Hit(TypedDict, total=False):
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float
    similarity: float          # ~ 1 - distance (cosine)
    embedding: List[float]     # present only if returned by Chroma

class HitWithoutEmb(TypedDict, total=False):
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float
    similarity: float          # ~ 1 - distance (cosine)


# ---------- Small utilities ----------

def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _to_list(x):
    if x is None:
        return []
    try:
        return x.tolist()  # numpy → python
    except Exception:
        return x

def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Make simple dicts operator-friendly for Chroma."""
    if not where:
        return None
    if len(where) == 1 and next(iter(where)).startswith("$"):
        return where
    clauses = []
    for k, v in where.items():
        if k.startswith("$"):
            return where
        if isinstance(v, dict) and any(str(op).startswith("$") for op in v):
            clauses.append({k: v})
        else:
            clauses.append({k: {"$eq": v}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

def _user_scope_filter(user_id: Optional[str], allow_public: bool) -> Optional[Dict[str, Any]]:
    if not user_id:
        return None
    return (
        {"$or": [{"user_id": {"$eq": user_id}}, {"visibility": {"$eq": "public"}}]}
        if allow_public else
        {"user_id": {"$eq": user_id}}
    )

def _merge_where(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    a = _normalize_where(a)
    b = _normalize_where(b)
    if a and b:
        return {"$and": [a, b]}
    return a or b


# ---------- Robust query helpers ----------

def _query_safe(q_emb: List[float], n_results: int, where: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Try several 'include' sets to be compatible across Chroma versions.
    Always includes 'documents'; adds others when supported.
    Returns (result, include_used).
    """
    include_candidates: List[List[str]] = [
        ["documents", "metadatas", "distances", "embeddings"],
        ["documents", "metadatas", "embeddings"],
        ["documents", "metadatas", "distances"],
        ["documents", "metadatas"],
        ["documents"],
    ]
    last_err: Optional[Exception] = None
    for inc in include_candidates:
        try:
            res = collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                include=inc,
                where=where or None,
            )
            return res, inc
        except ValueError as e:
            last_err = e
            continue
    # Final fallback: let Chroma decide defaults
    try:
        res = collection.query(query_embeddings=[q_emb], n_results=n_results, where=where or None)
        return res, []
    except Exception as e:
        raise last_err or e

def _build_hits(res: Dict[str, Any]) -> List[Hit]:
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0] if res.get("metadatas") else [{} for _ in docs]
    dists = (res.get("distances") or [[]])[0] if res.get("distances") else []
    embs  = (res.get("embeddings") or [[]])[0] if res.get("embeddings") else []
    ids   = (res.get("ids") or [[]])[0] if res.get("ids") else [""] * len(docs)

    dists = _to_list(dists)
    embs = _to_list(embs)

    hits: List[Hit] = []
    for i, doc in enumerate(docs):
        dist = float(dists[i]) if i < len(dists) else 1.0
        sim = 1.0 - dist if i < len(dists) else 0.0

        emb_i = None
        if i < len(embs):
            emb_i = _to_list(embs[i])
            if emb_i is not None:
                emb_i = [float(v) for v in emb_i]

        h: Hit = {
            "id": ids[i] if i < len(ids) else "",
            "document": doc,
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dist,
            "similarity": sim,
        }
        if emb_i is not None:
            h["embedding"] = emb_i
        hits.append(h)
    return hits

def _mmr_select(query_emb: List[float], hits: List[Hit], k: int, lam: float) -> List[Hit]:
    """MMR: balance relevance vs. diversity."""
    cand = [h for h in hits if "embedding" in h]
    if not cand:
        return sorted(hits, key=lambda h: h.get("similarity", 0.0), reverse=True)[:k]

    cand.sort(key=lambda h: _cos(query_emb, h["embedding"]), reverse=True)
    selected: List[Hit] = [cand.pop(0)]

    while cand and len(selected) < k:
        best_idx, best_score = 0, -1e9
        for j, h in enumerate(cand):
            rel = _cos(query_emb, h["embedding"])
            div = max(_cos(h["embedding"], s["embedding"]) for s in selected)
            score = lam * rel - (1.0 - lam) * div
            if score > best_score:
                best_score, best_idx = score, j
        selected.append(cand.pop(best_idx))

    return selected

def _mmr_select_without_emb(query_emb: List[float], hits: List[HitWithoutEmb], k: int, lam: float) -> List[HitWithoutEmb]:
    """MMR: balance relevance vs. diversity."""
    cand = [h for h in hits if "embedding" in h]
    if not cand:
        return sorted(hits, key=lambda h: h.get("similarity", 0.0), reverse=True)[:k]

    cand.sort(key=lambda h: _cos(query_emb, h["embedding"]), reverse=True)
    selected: List[HitWithoutEmb] = [cand.pop(0)]

    while cand and len(selected) < k:
        best_idx, best_score = 0, -1e9
        for j, h in enumerate(cand):
            rel = _cos(query_emb, h["embedding"])
            div = max(_cos(h["embedding"], s["embedding"]) for s in selected)
            score = lam * rel - (1.0 - lam) * div
            if score > best_score:
                best_score, best_idx = score, j
        selected.append(cand.pop(best_idx))

    return selected


# ---------- Public API ----------

def retrieve_hits(
    query: str,
    k: int = 5,
    fetch_k: int = 32,
    where: Optional[Dict[str, Any]] = None,
    distance_cutoff: Optional[float] = 0.35,  # cosine distance (0=identical; 1=unrelated)
    use_mmr: bool = True,
    mmr_lambda: float = 0.5,
    user_id: Optional[str] = None,
    allow_public: bool = True,
) -> List[Hit]:
    q_emb = get_query_embedding(query)
    combined_where = _merge_where(where, _user_scope_filter(user_id, allow_public))
    res, used_inc = _query_safe(q_emb, fetch_k, where=combined_where)
    hits = _build_hits(res)

    inc_has_dists = isinstance(used_inc, list) and ("distances" in used_inc)
    if (not inc_has_dists) and (not use_mmr) and any("embedding" in h for h in hits):
        # compute similarity if distances missing
        for h in hits:
            h["similarity"] = _cos(q_emb, h["embedding"]) if "embedding" in h else 0.0

    if distance_cutoff is not None and any("distance" in h for h in hits):
        hits = [h for h in hits if h.get("distance", 1.0) <= distance_cutoff]

    if not hits:
        return []

    if use_mmr and any("embedding" in h for h in hits) and len(hits) > k:
        selected = _mmr_select(q_emb, hits, k=k, lam=mmr_lambda)
    else:
        selected = sorted(hits, key=lambda h: h.get("similarity", 0.0), reverse=True)[:k]

    return selected


def retrieve_hits_without_emb(
    query: str,
    k: int = 5,
    fetch_k: int = 32,
    where: Optional[Dict[str, Any]] = None,
    distance_cutoff: Optional[float] = 0.35,  # cosine distance (0=identical; 1=unrelated)
    use_mmr: bool = True,
    mmr_lambda: float = 0.5,
    user_id: Optional[str] = None,
    allow_public: bool = True,
) -> List[HitWithoutEmb]:
    q_emb = get_query_embedding(query)
    combined_where = _merge_where(where, _user_scope_filter(user_id, allow_public))
    res, used_inc = _query_safe(q_emb, fetch_k, where=combined_where)
    hits = _build_hits(res)

    inc_has_dists = isinstance(used_inc, list) and ("distances" in used_inc)
    if (not inc_has_dists) and (not use_mmr) and any("embedding" in h for h in hits):
        # compute similarity if distances missing
        for h in hits:
            h["similarity"] = _cos(q_emb, h["embedding"]) if "embedding" in h else 0.0

    if distance_cutoff is not None and any("distance" in h for h in hits):
        hits = [h for h in hits if h.get("distance", 1.0) <= distance_cutoff]

    if not hits:
        return []

    if use_mmr and any("embedding" in h for h in hits) and len(hits) > k:
        selected = _mmr_select(q_emb, hits, k=k, lam=mmr_lambda)
    else:
        selected = sorted(hits, key=lambda h: h.get("similarity", 0.0), reverse=True)[:k]

    return selected


# ---------- Packing & budgets ----------

def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)  # ~4 chars/token

def messages_token_usage(msgs: list[dict]) -> int:
    overhead = 10
    return sum(approx_tokens(m.get("content", "")) + overhead for m in msgs)

def compute_context_budget(
    current_messages: list[dict],
    total_window: int = 4096,
    reserve_for_model_output: int = 700,
    hard_min: int = 400,
    hard_max: int = 2400,
) -> int:
    used = messages_token_usage(current_messages)
    remaining = max(0, total_window - used - reserve_for_model_output)
    if remaining <= 0:
        return 0
    return max(hard_min, min(hard_max, remaining))

def pack_context(hits: List[Hit], budget_tokens: int, header: str = "") -> str:
    if budget_tokens <= 0 or not hits:
        return header.strip()

    blocks: List[str] = []
    used = approx_tokens(header)
    for h in hits:
        meta = h.get("metadata", {}) or {}
        label = f"{meta.get('filename', 'memo')}#{meta.get('chunk', 0)}"
        head = f"[{label} | dist={h.get('distance', 1.0):.3f}]"
        block = f"{head}\n{(h.get('document') or '').strip()}\n"
        need = approx_tokens(block)
        if used + need > budget_tokens:
            break
        blocks.append(block)
        used += need

    out = (header + "\n" if header else "") + "\n".join(blocks)
    return out.strip()

def format_context(hits: List[Hit], max_chars: int = 4000) -> str:
    out: List[str] = []
    for h in hits:
        meta = h.get("metadata", {}) or {}
        label = f"{meta.get('filename', 'memo')}#{meta.get('chunk', 0)}"
        head = f"--- source: {label} | path: {meta.get('path', '-') } | dist: {h.get('distance', 1.0):.3f} ---"
        out.extend([head, (h.get("document") or "").strip(), ""])
        if sum(len(s) + 1 for s in out) > max_chars:
            break
    return "\n".join(out).strip()

def retrieve_relevant_chunks(query: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[str]:
    hits = retrieve_hits(query, k=k, where=where)
    return [h["document"] for h in hits]
