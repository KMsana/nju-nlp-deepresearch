"""Query-aware document chunking — replaces naive text[:5000] truncation."""

from typing import List


class QueryAwareChunker:
    """Split documents into overlapping chunks; select chunks most relevant to the query."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000,
                   overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _score_chunk(query: str, chunk: str) -> float:
        q_terms = set(query.lower().split())
        if not q_terms:
            return 0.0
        d_terms = set(chunk.lower().split())
        return len(q_terms & d_terms)

    @classmethod
    def select_relevant_chunks(cls, query: str, text: str,
                               max_chunks: int = 5,
                               chunk_size: int = 1000,
                               overlap: int = 200) -> str:
        chunks = cls.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if len(chunks) <= max_chunks:
            return text

        scored = [(cls._score_chunk(query, c), i, c)
                  for i, c in enumerate(chunks)]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:max_chunks]
        selected.sort(key=lambda x: x[1])  # restore original order
        return "\n".join(s[2] for s in selected)

    @classmethod
    def chunk_for_context(cls, query: str, text: str,
                          max_total_chars: int = 5000) -> str:
        max_chunks = max(1, max_total_chars // 1000)
        result = cls.select_relevant_chunks(query, text, max_chunks=max_chunks)
        if len(result) > max_total_chars:
            result = result[:max_total_chars]
        return result
