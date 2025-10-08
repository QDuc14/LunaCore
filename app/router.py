from __future__ import annotations

import os
import asyncio
import time
import re

from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime, timezone, timedelta, tzinfo
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from app.services.llm_provider import chat_once, chat_stream
from app.rag.retriever import retrieve_hits, pack_context, compute_context_budget, retrieve_hits_without_emb
from app.rag.embedder import (
    CHROMA_PATH,
    embed_and_store,
    collection,
    chroma_client,
    _cached_query_embedding,
)
from app.memory.store import memory
from app.memory.summarizer import update_summary
from app.memory.auto import retrieve_memories, save_memories_from_summary

router = APIRouter()

# =========================
# Config (from .env; safe defaults)
# =========================

load_dotenv(override=False)

AUTO_INGEST_EVERY_TURN  = os.getenv("LUNA_AUTO_INGEST", "1") != "0"   # save long-term memories each turn

STM_ENABLED             = os.getenv("LUNA_STM_ENABLED", "1") != "0"    # short-term memory default
STM_TOKENS_BUDGET       = int(os.getenv("LUNA_STM_TOKENS", "900"))

SUMMARY_EVERY_TURN      = os.getenv("LUNA_SUMMARY_EVERY_TURN", "1") != "0"
SUMMARY_KEEP_TURNS      = int(os.getenv("LUNA_SUMMARY_KEEP_TURNS", "4"))

DEFAULT_TIMEZONE        = os.getenv("LUNA_DEFAULT_TZ", "Asia/Ho_Chi_Minh")

PREFIXES                = os.getenv("LUNA_CMD_PREFIXES","")
CMDS_ENABLED            = os.getenv("LUNA_CMDS_ENABLED", "0") != "0"


# =========================
# Models
# =========================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    conversation_id: Optional[str] = None
    # If None, we fall back to LUNA_STM_ENABLED from env
    use_server_memory: Optional[bool] = None
    user_tz: Optional[str] = None

class IngestDoc(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestRequest(BaseModel):
    documents: Optional[List[str]] = None
    metadatas: Optional[List[Dict[str, Any]]] | Optional[Dict[str, Any]] = None
    items: Optional[List[IngestDoc]] = None
    user_id: Optional[str] = None
    chunk: bool = True
    chunk_size: int = 1200
    overlap: int = 200
    id_fields: Optional[List[str]] = ["path", "chunk", "user_id"]

class IngestResponse(BaseModel):
    received_docs: int
    added_docs: int
    skipped_duplicates: int

class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    dry_run: bool = False
    page_size: int = 500

class DeleteResponse(BaseModel):
    method: str
    matched: int
    deleted: int
    details: Dict[str, Any] = {}

class StatsResponse(BaseModel):
    collection: str
    metric: Optional[str] = None
    path: Optional[str] = None
    count: int
    vector_dim: Optional[int] = None
    samples: List[Dict[str, Any]] = Field(default_factory=list)

# =========================
# Helpers
# =========================
@lru_cache(maxsize=1)
def _persona_text() -> str:
    try:
        with open("luna_orgPesona1.txt", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "You are Luna, a helpful, concise assistant. Use provided context and cite as [filename#chunk]."

def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not chunk_size or chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, max(0, chunk_size - 1)))
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks

def _tolist(x):
    try:
        return x.tolist()
    except Exception:
        return x

def _normalize_where(where: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not where:
        return None
    if len(where) == 1 and next(iter(where)).startswith("$"):
        return where
    clauses = []
    for k, v in where.items():
        if k.startswith("$"):
            return where
        if isinstance(v, dict) and any(str(op).startswith("$") for op in v.keys()):
            clauses.append({k: v})
        else:
            clauses.append({k: {"$eq": v}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

def _count_by_where(where: Dict[str, Any], page_size: int = 500) -> int:
    total, offset = 0, 0
    where = _normalize_where(where)
    while True:
        try:
            got = collection.get(where=where, include=["metadatas"], limit=page_size, offset=offset)
        except Exception:
            got = collection.get(where=where, limit=page_size, offset=offset)
        metas = got.get("metadatas") or []
        batch = metas[0] if metas and isinstance(metas[0], list) else metas
        n = len(batch)
        if n == 0:
            break
        total += n
        offset += page_size
    return total

def _count_by_where_document(where_document: Dict[str, Any], page_size: int = 500) -> int:
    total, offset = 0, 0
    while True:
        try:
            got = collection.get(where_document=where_document, include=["metadatas"], limit=page_size, offset=offset)
        except Exception:
            got = collection.get(where_document=where_document, limit=page_size, offset=offset)
        metas = got.get("metadatas") or []
        batch = metas[0] if metas and isinstance(metas[0], list) else metas
        n = len(batch)
        if n == 0:
            break
        total += n
        offset += page_size
    return total

# ---- Time awareness (robust fallbacks) ----
def _resolve_tz(tz_name: Optional[str]) -> tzinfo:
    candidates = []
    if tz_name and tz_name.strip():
        candidates.append(tz_name.strip())
    if DEFAULT_TIMEZONE and DEFAULT_TIMEZONE not in candidates:
        candidates.append(DEFAULT_TIMEZONE)
    for z in ("Etc/UTC", "UTC"):
        if z not in candidates:
            candidates.append(z)
    for key in candidates:
        try:
            return ZoneInfo(key)
        except Exception:
            pass
    return timezone.utc

def _utc_offset_str(dt: datetime) -> str:
    off = dt.utcoffset() or timedelta(0)
    total_minutes = int(off.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hh, mm = divmod(total_minutes, 60)
    return f"UTC{sign}{hh:02d}:{mm:02d}"

def _time_system_message(tz_name: Optional[str]) -> Dict[str, str]:
    tz = _resolve_tz(tz_name)
    now = datetime.now(tz)
    offset = _utc_offset_str(now)
    tz_label = getattr(tz, "key", "UTC")
    content = (
        f"Current time context:\n"
        f"- Now (ISO): {now.isoformat()}\n"
        f"- Local tz: {tz_label} ({offset})\n"
        f"- Today: {now.strftime('%A, %B %d, %Y')}\n"
        f"Use this timezone for words like today/tomorrow/yesterday and for any date math."
    )
    return {"role": "system", "content": content}

def _use_stm(req_value: Optional[bool]) -> bool:
    # If request explicitly sets it, use that; otherwise use env default.
    return STM_ENABLED if req_value is None else bool(req_value)


# ---- command support (env: LUNA_CMDS_ENABLED, LUNA_CMD_PREFIXES) ----

_cmd_rx = re.compile(rf"^\s*(?P<prefix>{'|'.join(re.escape(p) for p in PREFIXES)})\s*(?P<body>.*)$") if PREFIXES else None

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

async def _ingest_texts(items: List[str], metadatas: List[Dict]) -> int:
    # embed_and_store is sync → offload
    return await asyncio.to_thread(
        embed_and_store, items, metadatas=metadatas, id_fields=["memory_key"]
    )

async def _handle_cmd(name: str, body: str, *, uid: str, cid: str) -> str:
    if not body:
        return f"'{name}:' needs some text. Example: {name}: your content here"

    now = _now_iso()
    if name in {"remember","kb","ingest"}:
        topic = "memory" if name == "remember" else ("kb" if name == "kb" else "ingest")
        # Use a stable key so re-ingests refresh rather than duplicate
        key = f"{uid}:{cid}:{topic}:{hash(body)}"
        metas = [{
            "topic": topic, "user_id": uid, "conversation_id": cid,
            "memory_key": key, "source": "inchat", "created_at": now, "ttl_days": 365
        }]
        await _ingest_texts([body], metas)
        if topic == "memory":
            memory.append(f"{uid}:{cid}", "system", f"[MEMO] {body}")
            memory.trim_to_budget(f"{uid}:{cid}", budget_tokens=STM_TOKENS_BUDGET)
        return "Noted." if topic == "memory" else "Added."

    if name == "note":
        memory.append(f"{uid}:{cid}", "user", f"[NOTE] {body}")
        memory.trim_to_budget(f"{uid}:{cid}", budget_tokens=STM_TOKENS_BUDGET)
        return "Okay Luna will take a note for this conversation!"

    if name == "correct":
        memory.append(f"{uid}:{cid}", "system", f"[CORRECTION] {body}")
        memory.trim_to_budget(f"{uid}:{cid}", budget_tokens=STM_TOKENS_BUDGET)
        metas = [{"topic":"correction","user_id":uid,"conversation_id":cid,"memory_key":f"{uid}:{cid}:cor:{hash(body)}","source":"inchat","created_at":now}]
        await _ingest_texts([body], metas)
        return "Got it. I’ve remember that correction."

    if name == "forget":
        # Simple tombstone approach; replace with a real delete if you expose one.
        metas = [{"topic":"tombstone","user_id":uid,"conversation_id":cid,"memory_key":f"{uid}:{cid}:tomb:{hash(body)}","created_at":now,"for_text_contains":body}]
        await _ingest_texts([f"TOMBSTONE: forget anything containing -> {body}"], metas)
        return "Luna will forget it..."

    return None  # not a command we handle

async def maybe_handle_inchat_command(text: Optional[str], *, uid: str, cid: str) -> Optional[str]:
    if not (CMDS_ENABLED and _cmd_rx and text):
        return None
    m = _cmd_rx.match(text)
    if not m:
        return None
    name = m.group("prefix").rstrip(":").lower()
    body = (m.group("body") or "").strip()
    return await _handle_cmd(name, body, uid=uid, cid=cid)


# =========================
# Endpoints
# =========================
@router.get("/health")
def health():
    try:
        n = collection.count()
        return {"ok": True, "vectorstore_count": n}
    except Exception:
        return {"ok": True, "vectorstore_count": None}

@router.get("/search")
def search_debug(q: str, k: int = 5, use_cutoff: bool = True, user_id: Optional[str] = None):
    hits = retrieve_hits_without_emb(
        q,
        k=k,
        distance_cutoff=(0.4 if use_cutoff else None),
        user_id=(user_id or "anon").strip() if user_id else None,
        allow_public=True,
        where=None,
    )
    return {"hits": hits}

@router.get("/stats", response_model=StatsResponse)
def stats(sample: int = 3):
    try:
        metric = None
        if hasattr(collection, "metadata") and isinstance(collection.metadata, dict):
            metric = collection.metadata.get("hnsw:space")
    except Exception:
        metric = None

    cnt = collection.count()
    dim = None
    try:
        pk = collection.peek(include=["embeddings"])
        embs = pk.get("embeddings") or [[]]
        if embs and len(embs[0]) > 0:
            first = _tolist(embs[0])
            dim = len(first) if isinstance(first, (list, tuple)) else None
    except Exception:
        pass

    samples: List[Dict[str, Any]] = []
    if sample > 0 and cnt > 0:
        got = collection.get(limit=sample, include=["metadatas"])
        metas_col = got.get("metadatas") or []
        metas = metas_col[0] if metas_col and isinstance(metas_col[0], list) else metas_col
        for m in metas[:sample]:
            mm = m or {}
            samples.append({
                "category": mm.get("category"),
                "tags": mm.get("tags"),
                "path": mm.get("path"),
                "chunk": mm.get("chunk"),
                "source": mm.get("source"),
            })

    return StatsResponse(
        collection="luna-memory",
        metric=metric,
        path=CHROMA_PATH,
        count=cnt,
        vector_dim=dim,
        samples=samples,
    )

@router.get("/cache/embeddings")
def cache_embeddings_stats():
    info = _cached_query_embedding.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
    }

@router.post("/cache/embeddings/clear")
def cache_embeddings_clear():
    from app.rag.embedder import clear_embedding_cache
    clear_embedding_cache()
    return {"status": "cleared"}

@router.get("/memory/get")
def memory_get(conversation_id: str = "default", user_id: Optional[str] = None):
    sid = f"{(user_id or 'anon').strip()}:{conversation_id}"
    return {"conversation_id": conversation_id, "user_id": user_id, "messages": memory.get(sid)}

@router.get("/memory/summary")
def memory_summary(conversation_id: str = "default", user_id: Optional[str] = None):
    sid = f"{(user_id or 'anon').strip()}:{conversation_id}"
    return {"conversation_id": conversation_id, "user_id": user_id, "summary": memory.get_summary(sid)}

@router.get("/memory/list")
def memory_list(conversation_id: str, user_id: Optional[str] = None, limit: int = 20, offset: int = 0):
    scope = (user_id or "").strip() or f"conv:{conversation_id}"
    where = {"$and":[
        {"topic":{"$eq":"memory"}},
        {"$or":[{"scope":{"$eq": scope}}, {"conversation_id":{"$eq": conversation_id}}]}
    ]}
    try:
        res = collection.get(where=where, include=["documents","metadatas"], limit=limit, offset=offset)
    except Exception:
        res = collection.get(where=where, limit=limit, offset=offset)

    docs_col = res.get("documents") or []
    metas_col = res.get("metadatas") or []
    docs  = docs_col[0]  if docs_col  and isinstance(docs_col[0],  list) else docs_col
    metas = metas_col[0] if metas_col and isinstance(metas_col[0], list) else metas_col

    items = []
    for d, m in zip(docs or [], metas or []):
        mm = m or {}
        items.append({
            "memory_key": mm.get("memory_key"),
            "scope": mm.get("scope"),
            "text": d,
            "expires_at": mm.get("expires_at"),
            "created_at": mm.get("created_at"),
            "ttl_days": mm.get("ttl_days"),
            "seen_count": mm.get("seen_count"),
        })
    return {"conversation_id": conversation_id, "user_id": user_id, "items": items}

@router.post("/memory/clear")
def memory_clear(conversation_id: str = "default", user_id: Optional[str] = None):
    sid = f"{(user_id or 'anon').strip()}:{conversation_id}"
    memory.set(sid, [])
    memory.set_summary(sid, "")
    return {"status": f"cleared {sid}"}

# ========== CHAT (env-driven STM; auto-ingest toggle) ==========
@router.post("/chat")
async def chat(request: ChatRequest):
    latest_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not latest_user_msg:
        raise HTTPException(status_code=400, detail="No user message found.")
    model_name = (request.model or "").strip()
    if model_name.lower() in {"", "string", "model", "<model>"}:
        raise HTTPException(status_code=400, detail="Provide a valid Ollama model tag in 'model'.")

    cid = request.conversation_id or "default"
    uid = (request.user_id or "anon").strip()
    sid = f"{uid}:{cid}"
    use_stm = _use_stm(request.use_server_memory)
    ack = await maybe_handle_inchat_command(latest_user_msg, uid=uid, cid=cid)
    if ack:
        return {"response": ack}

    luna_persona = {"role": "system", "content": _persona_text()}
    citation_rule = {"role": "system", "content": "When you use provided context, cite the source label like [filename#chunk]."}
    time_msg = _time_system_message(request.user_tz)
    user_id_msg = {"role": "system", "content": f"User: {request.user_name} (id: {uid})."} if request.user_name else {"role": "system", "content": f"User id: {uid}."}

    history_msgs: List[dict] = []
    if use_stm:
        memory.append(sid, "user", latest_user_msg)
        memory.trim_to_budget(sid, budget_tokens=STM_TOKENS_BUDGET)
        history_msgs = memory.get(sid)

    # Long-term memory recall
    memory_snippets = retrieve_memories(latest_user_msg, conversation_id=cid, user_id=uid, k=4)
    memory_msg = {"role": "system", "content": "Relevant prior facts:\n" + "\n".join(f"- {m}" for m in memory_snippets)} if memory_snippets else None

    base_msgs = [m for m in [luna_persona, citation_rule, time_msg, memory_msg, user_id_msg] if m] + history_msgs + [m.dict() for m in request.messages]
    budget = compute_context_budget(base_msgs, total_window=4096, reserve_for_model_output=700)

    q = latest_user_msg.strip()
    do_rag = len(q) >= 12
    fetch_k = 16 if len(q) < 120 else 32
    top_k   = 5  if len(q) < 120 else 8
    hits = retrieve_hits(q, k=top_k, fetch_k=fetch_k, distance_cutoff=0.4, user_id=uid, allow_public=True, where=None) if do_rag else []
    context_text = pack_context(hits, budget_tokens=budget, header="Use the following context to answer the user:") if hits else ""

    MAX_MSGS = 30
    pruned = [m.dict() for m in request.messages][-MAX_MSGS:]
    enriched_messages = [m for m in [luna_persona, citation_rule, time_msg, memory_msg, user_id_msg] if m]
    if context_text:
        enriched_messages.append({"role": "system", "content": context_text})
    enriched_messages += history_msgs + [m.dict() for m in request.messages]

    prompt_tokens_est = sum(len((m.get("content") or "")) // 4 for m in enriched_messages)
    reply = await chat_once(enriched_messages, model=model_name)

    # STM append
    if use_stm:
        memory.append(sid, "assistant", reply)
        memory.trim_to_budget(sid, budget_tokens=STM_TOKENS_BUDGET)

    # Rolling summary + env-controlled auto-ingest
    if SUMMARY_EVERY_TURN or AUTO_INGEST_EVERY_TURN:
        turn_msgs = history_msgs + [m.dict() for m in request.messages] + [{"role": "assistant", "content": reply}]
        prior_summary = memory.get_summary(sid) if use_stm else ""
        new_summary = await update_summary(prior_summary, turn_msgs, model=model_name)
        if use_stm:
            memory.set_summary(sid, new_summary)
            if SUMMARY_KEEP_TURNS >= 0:
                memory.set(sid, (history_msgs + [{"role": "assistant", "content": reply}])[-SUMMARY_KEEP_TURNS:])
        if AUTO_INGEST_EVERY_TURN and new_summary:
            try:
                save_memories_from_summary(new_summary, conversation_id=cid, user_id=uid, ttl_days=30)
            except Exception:
                pass

    return {"response": reply}

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    latest_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not latest_user_msg:
        raise HTTPException(status_code=400, detail="No user message found.")
    model_name = (request.model or "").strip()
    if model_name.lower() in {"", "string", "model", "<model>"}:
        raise HTTPException(status_code=400, detail="Provide a valid Ollama model tag in 'model'.")

    cid = request.conversation_id or "default"
    uid = (request.user_id or "anon").strip()
    sid = f"{uid}:{cid}"
    use_stm = _use_stm(request.use_server_memory)
    ack = await maybe_handle_inchat_command(latest_user_msg, uid=uid, cid=cid)
    if ack:
        async def event_stream():
            yield f"data: {ack}\n\n".encode("utf-8")
            yield b"event: done\ndata: true\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    luna_persona = {"role": "system", "content": _persona_text()}
    citation_rule = {"role": "system", "content": "When you use provided context, cite [filename#chunk]."}
    time_msg = _time_system_message(request.user_tz)
    user_id_msg = {"role": "system", "content": f"User: {request.user_name} (id: {uid})."} if request.user_name else {"role": "system", "content": f"User id: {uid}."}

    history_msgs: List[dict] = []
    if use_stm:
        memory.append(sid, "user", latest_user_msg)
        memory.trim_to_budget(sid, budget_tokens=STM_TOKENS_BUDGET)
        history_msgs = memory.get(sid)

    memory_snippets = retrieve_memories(latest_user_msg, conversation_id=cid, user_id=uid, k=4)
    memory_msg = {"role": "system", "content": "Relevant prior facts:\n" + "\n".join(f"- {m}" for m in memory_snippets)} if memory_snippets else None

    base_msgs = [m for m in [luna_persona, citation_rule, time_msg, memory_msg, user_id_msg] if m] + history_msgs + [m.dict() for m in request.messages]
    budget = compute_context_budget(base_msgs, total_window=4096, reserve_for_model_output=700)

    q = latest_user_msg.strip()
    do_rag = len(q) >= 12
    fetch_k = 16 if len(q) < 120 else 32
    top_k   = 5  if len(q) < 120 else 8
    hits = retrieve_hits(q, k=top_k, fetch_k=fetch_k, distance_cutoff=0.4, user_id=uid, allow_public=True, where=None) if do_rag else []
    context_text = pack_context(hits, budget_tokens=budget, header="Use the following context to answer the user:") if hits else ""

    enriched_messages = [m for m in [luna_persona, citation_rule, time_msg, memory_msg, user_id_msg] if m]
    if context_text:
        enriched_messages.append({"role": "system", "content": context_text})
    
    MAX_MSGS = 30
    pruned = [m.dict() for m in request.messages][-MAX_MSGS:]
    enriched_messages += history_msgs + [m.dict() for m in request.messages]

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            collected: List[str] = []

            async for chunk in chat_stream(enriched_messages, model=model_name):
                collected.append(chunk)
                yield f"data: {chunk}\n\n".encode("utf-8")
            # after loop:
            text = "".join(collected)

            # 1) Tell the client we're done ASAP (don't block on post-processing)
            yield f"event: done\ndata: {text}\n\n".encode("utf-8")

            # 2) Post-processing in background
            async def _post_process():
                try:
                    # Short-term memory (if enabled)
                    if use_stm:
                        memory.append(sid, "assistant", text)
                        memory.trim_to_budget(sid, budget_tokens=STM_TOKENS_BUDGET)

                    if SUMMARY_EVERY_TURN or AUTO_INGEST_EVERY_TURN:
                        # Build the turn messages once
                        turn_msgs = (
                            history_msgs
                            + [m.dict() for m in request.messages]
                            + [{"role": "assistant", "content": text}]
                        )

                        # If update_summary is synchronous/blocking, offload it
                        prior_summary = memory.get_summary(sid) if use_stm else ""
                        # new_summary = await asyncio.to_thread(
                        #     update_summary, prior_summary, turn_msgs, model_name
                        # )
                        new_summary = await update_summary(prior_summary, turn_msgs, model_name)

                        if use_stm:
                            memory.set_summary(sid, new_summary)
                            if SUMMARY_KEEP_TURNS >= 0:
                                memory.set(
                                    sid,
                                    (history_msgs + [{"role": "assistant", "content": text}])[-SUMMARY_KEEP_TURNS:]
                                )

                        if AUTO_INGEST_EVERY_TURN and new_summary:
                            # save_memories_from_summary is sync -> offload to thread
                            await asyncio.to_thread(
                                save_memories_from_summary, new_summary, cid, uid, 30
                            )
                except Exception as e:
                    # Log, but never break the stream
                    import logging
                    logging.getLogger("luna").exception("post_process failed: %s", e)

            # Fire and forget
            asyncio.create_task(_post_process())

        except Exception as e:
            msg = str(e).replace("\n", " ").strip()
            yield f"event: error\ndata: {msg}\n\n".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ========== DISCORD STREAM (no citation rule) ==========
@router.post("/discord/chat/stream")
async def discord_chat_stream(request: ChatRequest):
    latest_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not latest_user_msg:
        raise HTTPException(status_code=400, detail="No user message found.")
    model_name = (request.model or "").strip()
    if model_name.lower() in {"", "string", "model", "<model>"}:
        raise HTTPException(status_code=400, detail="Provide a valid Ollama model tag in 'model'.")

    cid = request.conversation_id or "default"
    uid = (request.user_id or "anon").strip()
    sid = f"{uid}:{cid}"
    use_stm = _use_stm(request.use_server_memory)
    ack = await maybe_handle_inchat_command(latest_user_msg, uid=uid, cid=cid)
    if ack:
        async def event_stream():
            yield f"data: {ack}\n\n".encode("utf-8")
            yield b"event: done\ndata: true\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    luna_persona = {"role": "system", "content": _persona_text()}
    time_msg = _time_system_message(request.user_tz)
    user_id_msg = {"role": "system", "content": f"User: {request.user_name} (id: {uid})."} if request.user_name else {"role": "system", "content": f"User id: {uid}."}

    history_msgs: List[dict] = []
    if use_stm:
        memory.append(sid, "user", latest_user_msg)
        memory.trim_to_budget(sid, budget_tokens=STM_TOKENS_BUDGET)
        history_msgs = memory.get(sid)

    memory_snippets = retrieve_memories(latest_user_msg, conversation_id=cid, user_id=uid, k=4)
    memory_msg = {"role": "system", "content": "Relevant prior facts:\n" + "\n".join(f"- {m}" for m in memory_snippets)} if memory_snippets else None

    base_msgs = [m for m in [luna_persona, time_msg, memory_msg, user_id_msg] if m] + history_msgs + [m.dict() for m in request.messages]
    budget = compute_context_budget(base_msgs, total_window=4096, reserve_for_model_output=700)

    q = latest_user_msg.strip()
    do_rag = len(q) >= 12
    fetch_k = 16 if len(q) < 120 else 32
    top_k   = 5  if len(q) < 120 else 8
    hits = retrieve_hits(q, k=top_k, fetch_k=fetch_k, distance_cutoff=0.4, user_id=uid, allow_public=True, where=None) if do_rag else []
    context_text = pack_context(hits, budget_tokens=budget, header="Use the following context to answer the user:") if hits else ""

    enriched_messages = [m for m in [luna_persona, time_msg, memory_msg, user_id_msg] if m]
    if context_text:
        enriched_messages.append({"role": "system", "content": context_text})
    
    MAX_MSGS = 30
    pruned = [m.dict() for m in request.messages][-MAX_MSGS:]
    enriched_messages += history_msgs + [m.dict() for m in request.messages]

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            collected: List[str] = []

            async for chunk in chat_stream(enriched_messages, model=model_name):
                collected.append(chunk)
                yield f"data: {chunk}\n\n".encode("utf-8")
            # after loop:
            text = "".join(collected)

            # 1) Tell the client we're done ASAP (don't block on post-processing)
            yield f"event: done\ndata: {text}\n\n".encode("utf-8")

            # 2) Post-processing in background
            async def _post_process():
                try:
                    # Short-term memory (if enabled)
                    if use_stm:
                        memory.append(sid, "assistant", text)
                        memory.trim_to_budget(sid, budget_tokens=STM_TOKENS_BUDGET)

                    if SUMMARY_EVERY_TURN or AUTO_INGEST_EVERY_TURN:
                        # Build the turn messages once
                        turn_msgs = (
                            history_msgs
                            + [m.dict() for m in request.messages]
                            + [{"role": "assistant", "content": text}]
                        )

                        # If update_summary is synchronous/blocking, offload it
                        prior_summary = memory.get_summary(sid) if use_stm else ""
                        new_summary = await update_summary(prior_summary, turn_msgs, model_name)

                        if use_stm:
                            memory.set_summary(sid, new_summary)
                            if SUMMARY_KEEP_TURNS >= 0:
                                memory.set(
                                    sid,
                                    (history_msgs + [{"role": "assistant", "content": text}])[-SUMMARY_KEEP_TURNS:]
                                )

                        if AUTO_INGEST_EVERY_TURN and new_summary:
                            # save_memories_from_summary is sync -> offload to thread
                            await asyncio.to_thread(
                                save_memories_from_summary, new_summary, cid, uid, 30
                            )
                except Exception as e:
                    # Log, but never break the stream
                    import logging
                    logging.getLogger("luna").exception("post_process failed: %s", e)

            # Fire and forget
            asyncio.create_task(_post_process())

        except Exception as e:
            msg = str(e).replace("\n", " ").strip()
            yield f"event: error\ndata: {msg}\n\n".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ========== Ingest / Delete / Drop ==========
@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
    now = datetime.now(timezone.utc).isoformat()
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    uid = (payload.user_id or "").strip() or None

    def add(text: str, meta: Dict[str, Any]):
        docs.append(text)
        metas.append(meta)

    if payload.items:
        for item in payload.items:
            base_meta = dict(item.metadata or {})
            if uid: base_meta["user_id"] = uid
            base_meta.setdefault("source", "api")
            base_meta.setdefault("created_at", now)
            if payload.chunk:
                for idx, ch in enumerate(_split_text(item.text, payload.chunk_size, payload.overlap)):
                    m = dict(base_meta); m.setdefault("chunk", idx)
                    add(ch, m)
            else:
                add(item.text, base_meta)
    else:
        documents = payload.documents
        if not documents:
            raise HTTPException(status_code=400, detail="Provide 'items' or 'documents'.")
        # Normalize metadatas to a per-doc list
        if payload.metadatas is None:
            meta_list = [{"source": "api", "created_at": now, **({"user_id": uid} if uid else {})} for _ in documents]
        elif isinstance(payload.metadatas, dict):
            base = dict(payload.metadatas)
            if uid: base["user_id"] = uid
            base.setdefault("source", "api"); base.setdefault("created_at", now)
            meta_list = [base for _ in documents]
        else:
            if len(payload.metadatas) != len(documents):
                raise HTTPException(status_code=400, detail="metadatas list must match documents length")
            meta_list = []
            for md in payload.metadatas:
                m = dict(md)
                if uid: m["user_id"] = uid
                m.setdefault("source", "api")
                m.setdefault("created_at", now)
                meta_list.append(m)
        for text, meta in zip(documents, meta_list):
            if payload.chunk:
                for idx, ch in enumerate(_split_text(text, payload.chunk_size, payload.overlap)):
                    mm = dict(meta); mm.setdefault("chunk", idx)
                    add(ch, mm)
            else:
                add(text, meta)

    received = len(docs)
    if received == 0:
        return IngestResponse(received_docs=0, added_docs=0, skipped_duplicates=0)

    added = embed_and_store(docs, metadatas=metas, id_fields=payload.id_fields)
    return IngestResponse(
        received_docs=received,
        added_docs=added,
        skipped_duplicates=max(0, received - added),
    )

@router.post("/delete", response_model=DeleteResponse)
def delete_embeddings(req: DeleteRequest):
    if req.ids:
        matched = len(req.ids)
        if req.dry_run:
            return DeleteResponse(method="ids", matched=matched, deleted=0, details={"note": "dry_run"})
        collection.delete(ids=req.ids)
        return DeleteResponse(method="ids", matched=matched, deleted=matched)

    if req.where:
        normalized = _normalize_where(req.where)
        matched = _count_by_where(normalized, page_size=req.page_size)
        if req.dry_run:
            return DeleteResponse(method="where", matched=matched, deleted=0, details={"note": "dry_run"})
        collection.delete(where=normalized)
        return DeleteResponse(method="where", matched=matched, deleted=matched)

    if req.where_document:
        matched = _count_by_where_document(req.where_document, page_size=req.page_size)
        if req.dry_run:
            return DeleteResponse(method="where_document", matched=matched, deleted=0, details={"note": "dry_run"})
        collection.delete(where_document=req.where_document)
        return DeleteResponse(method="where_document", matched=matched, deleted=matched)

    raise HTTPException(status_code=400, detail="Provide one of: ids | where | where_document")

@router.post("/collections/drop")
def drop_collection(name: str = "luna-memory"):
    chroma_client.delete_collection(name)
    return {"status": f"Collection '{name}' deleted."}
