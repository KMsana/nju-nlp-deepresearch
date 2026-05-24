"""Two-stage retrieval: BM25 top-15 → re-rank to top-5 via term overlap."""

from typing import Any, Dict, List


class ReRanker:
    """Fetch top-N from BM25, re-rank with query-term overlap to select top-K."""

    @staticmethod
    def _normalize_scores(results: List[Dict]) -> List[float]:
        scores = [r["score"] for r in results]
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    @staticmethod
    def _term_overlap(query: str, text: str) -> float:
        q_terms = set(query.lower().split())
        if not q_terms:
            return 0.0
        d_terms = set(text.lower().split())
        return len(q_terms & d_terms) / len(q_terms)

    @classmethod
    def rerank(cls, query: str, results: List[Dict],
               top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
        if not results:
            return []
        norm = cls._normalize_scores(results)
        for i, r in enumerate(results):
            overlap = cls._term_overlap(query, r.get("text", ""))
            r["rerank_score"] = alpha * norm[i] + (1 - alpha) * overlap
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
