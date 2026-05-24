"""Structured evidence storage — replaces Mem.findings: List[str]."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Evidence:
    fact: str
    docid: str
    query: str
    round: int
    confidence: str = "medium"
    source_quote: str = ""


class EvidenceStore:
    def __init__(self):
        self._items: List[Evidence] = []
        self._fact_set: set = set()
        self._ruled_out: List[Dict] = []

    # ── add ──────────────────────────────────────────────────────

    def add(self, fact: str, docid: str = "", query: str = "",
            round_num: int = 0, confidence: str = "medium",
            source_quote: str = "") -> bool:
        key = fact.strip().lower()
        if not key or key in self._fact_set:
            return False
        self._fact_set.add(key)
        self._items.append(Evidence(
            fact=fact.strip(), docid=docid, query=query,
            round=round_num, confidence=confidence,
            source_quote=source_quote,
        ))
        return True

    def add_batch(self, evidence_list: List[Dict],
                  query: str = "", round_num: int = 0) -> int:
        added = 0
        for e in evidence_list:
            if not isinstance(e, dict) or "fact" not in e:
                continue
            if self.add(
                fact=e["fact"],
                docid=e.get("docid", ""),
                query=query,
                round_num=round_num,
                confidence=e.get("confidence", "medium"),
                source_quote=e.get("source_quote", ""),
            ):
                added += 1
        return added

    # ── query ────────────────────────────────────────────────────

    def get_all(self) -> List[Evidence]:
        return list(self._items)

    def get_by_round(self, round_num: int) -> List[Evidence]:
        return [e for e in self._items if e.round == round_num]

    def get_high_confidence(self) -> List[Evidence]:
        return [e for e in self._items if e.confidence == "high"]

    def has_fact(self, fact: str) -> bool:
        return fact.strip().lower() in self._fact_set

    def count(self) -> int:
        return len(self._items)

    # ── ruled-out candidates ─────────────────────────────────────

    def add_ruled_out(self, candidate: str, reason: str = "") -> None:
        c = candidate.strip()
        if c and c not in {r["candidate"] for r in self._ruled_out}:
            self._ruled_out.append({"candidate": c, "reason": reason})

    def ruled_out_summary(self) -> str:
        if not self._ruled_out:
            return "(none)"
        return "\n".join(f"- {r['candidate']}: {r['reason']}"
                         for r in self._ruled_out[-5:])

    # ── context formatting ───────────────────────────────────────

    def summary_for_context(self, max_items: int = 10) -> str:
        if not self._items:
            return "(no evidence collected yet)"
        recent = self._items[-max_items:]
        lines = []
        for e in recent:
            src = f" (doc: {e.docid})" if e.docid else ""
            lines.append(f"- [{e.confidence}]{src} {e.fact}")
        return "\n".join(lines)

    def full_summary(self) -> str:
        parts = [f"## Collected Evidence ({self.count()} items)"]
        parts.append(self.summary_for_context(max_items=50))
        if self._ruled_out:
            parts.append(f"\n## Ruled Out\n{self.ruled_out_summary()}")
        return "\n".join(parts)
