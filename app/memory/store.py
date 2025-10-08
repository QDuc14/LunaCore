from __future__ import annotations
from typing import List, Dict

# Simple in-process short-term memory store with a rolling summary slot.
# Not persisted; keep it tiny with a token budget.

def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)

class _MemoryStore:
    def __init__(self):
        self._msgs: Dict[str, List[dict]] = {}
        self._summary: Dict[str, str] = {}

    # Messages
    def get(self, key: str) -> List[dict]:
        return list(self._msgs.get(key, []))

    def set(self, key: str, msgs: List[dict]) -> None:
        self._msgs[key] = list(msgs)

    def append(self, key: str, role: str, content: str) -> None:
        self._msgs.setdefault(key, []).append({"role": role, "content": content})

    def trim_to_budget(self, key: str, budget_tokens: int = 900) -> None:
        msgs = self._msgs.get(key, [])
        if not msgs:
            return
        # drop from the oldest until under budget
        while msgs and sum(_approx_tokens(m.get("content", "")) + 10 for m in msgs) > budget_tokens:
            msgs.pop(0)
        self._msgs[key] = msgs

    # Summary
    def get_summary(self, key: str) -> str:
        return self._summary.get(key, "")

    def set_summary(self, key: str, summary: str) -> None:
        self._summary[key] = summary

memory = _MemoryStore()
