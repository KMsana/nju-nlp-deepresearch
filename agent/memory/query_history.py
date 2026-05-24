"""Sliding-window query history — replaces permanent Mem.seen() dedupe."""

from typing import List


class QueryHistory:
    def __init__(self, window_size: int = 2):
        self._queries: List[str] = []
        self._window_size = window_size

    def add(self, query: str) -> None:
        q = query.strip()
        if q:
            self._queries.append(q)

    def was_recent(self, query: str) -> bool:
        """True if query matches any of the last window_size entries (case-insensitive).

        This is the PRIMARY dedupe gate. A→B→A is allowed (window=2 means
        only the last 2 queries are blocked, so after B is searched, A
        can be searched again).
        """
        q = query.strip().lower()
        if not q:
            return False
        recent = self._queries[-self._window_size:]
        return any(q == r.lower() for r in recent)

    def was_ever_searched(self, query: str) -> bool:
        """Global lookup — only for fallback logic, NOT for primary dedupe."""
        q = query.strip().lower()
        return any(q == prev.lower() for prev in self._queries)

    def all_queries(self) -> List[str]:
        return list(self._queries)

    def recent_queries(self, n: int = 5) -> List[str]:
        return self._queries[-n:]

    def count(self) -> int:
        return len(self._queries)
