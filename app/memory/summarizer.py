from __future__ import annotations
from typing import List

from app.services.llm_provider import chat_once

SYSTEM = (
    "You are a careful conversation summarizer. "
    "Summarize the dialogue into 3–6 terse bullet points that capture decisions, facts, names, and corrections. "
    "Do NOT include chit-chat. Prefer stable facts over transient phrasing. "
    "Output plain bullets (one per line) with no prefix except '- '."
)

def _render_history(turns: List[dict]) -> str:
    out = []
    for t in turns[-16:]:  # cap
        role = t.get("role","")
        content = (t.get("content","") or "").strip()
        if not content: 
            continue
        out.append(f"{role.upper()}: {content}")
    return "\n".join(out)

async def _call(model: str, messages: List[dict]) -> str:
    return await chat_once(messages, model=model)

async def update_summary(prior: str, turns: List[dict], model: str) -> str:
    hist = _render_history(turns)
    user_prompt = (
        f"Existing summary (may be empty):\n{prior or '(none)'}\n\n"
        f"Recent dialogue (latest last):\n{hist}\n\n"
        "Update the summary bullets. Keep only stable facts useful for future turns. "
        "Return 3–6 lines, each starting with '- '."
    )
    return await _call(model, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ])
