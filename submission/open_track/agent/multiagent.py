import json
import re
from typing import Any, Dict, List, Tuple

def _chat(client, model, msgs, max_tok=6000):
    try:
        r = client.simple_chat(model=model, messages=msgs, temperature=0.0, max_tokens=max_tok)
        return r["choices"][0]["message"]["content"]
    except Exception as e: return f"ERROR: {e}"

def _strip_think(text: str) -> str:
    t = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    t = re.sub(r'<think>.*$', '', t, flags=re.DOTALL).strip()
    return t if t else text.strip()


_STOPWORDS = {
    'the','a','an','is','was','are','were','be','been','being','in','on','at',
    'to','for','of','from','by','with','and','or','but','not','this','that',
    'these','those','can','you','tell','find','what','when','where','who','how',
    'why','which','name','one','first','last','mid','there','their','they','them',
    'has','have','had','its','also','about','after','before','during','into',
    'over','under','between','among','whose','whom','did','does','do','known',
    'called','would','could','should','answer','question'
}


def _norm_token(tok: str) -> str:
    tok = tok.lower().strip("'")
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and len(tok) > 4 and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> List[str]:
    return [t.lower().strip("'") for t in re.findall(r"[A-Za-z0-9]+", text or "")]


def _norm_tokens(text: str) -> List[str]:
    return [_norm_token(t) for t in _tokens(text)]


def _content_terms(text: str, max_terms: int = 12) -> List[str]:
    terms, seen = [], set()
    for tok in _tokens(text):
        if tok in _STOPWORDS:
            continue
        if len(tok) < 3 and not tok.isdigit():
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


def _extract_anchors(text: str, max_terms: int = 6) -> List[str]:
    anchors, seen = [], set()
    quoted = re.findall(r'"([^"]{2,80})"', text or "")
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b', text or "")
    years = re.findall(r'\b(?:1[7-9]\d{2}|20[0-2]\d)s?\b', text or "")
    for chunk in quoted + caps + years:
        for term in _content_terms(chunk, max_terms=4):
            if term not in seen:
                seen.add(term)
                anchors.append(term)
        if len(anchors) >= max_terms:
            break
    return anchors[:max_terms]


def _clean_query(q: str, min_terms: int = 2, max_terms: int = 6) -> str:
    """把自然语言/LLM输出压缩成适合 BM25 的短关键词查询。"""
    q = re.sub(r'(?i)^\s*(search query|query|keywords?)\s*:\s*', '', q or "")
    terms, seen = [], set()
    for tok in _tokens(q):
        if tok in _STOPWORDS:
            continue
        if len(tok) < 3 and not tok.isdigit():
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
        if len(terms) >= max_terms:
            break
    return " ".join(terms) if len(terms) >= min_terms else ""


def _clean_queries(queries: List[str], limit: int = 4) -> List[str]:
    cleaned, seen = [], set()
    for q in queries:
        cq = _clean_query(q)
        if not cq:
            continue
        key = " ".join(_tokens(cq))
        if key not in seen:
            seen.add(key)
            cleaned.append(cq)
        if len(cleaned) >= limit:
            break
    return cleaned


def _query_key(q: str) -> str:
    return " ".join(_norm_tokens(q))


_RELATION_HINTS = {
    "author","writer","novelist","poet","playwright","composer","director",
    "actor","actress","singer","artist","producer","editor","publisher",
    "founder","cofounder","president","minister","governor","mayor","senator",
    "coach","manager","captain","player","professor","teacher","librarian",
    "scientist","inventor","architect","engineer","spouse","wife","husband",
    "partner","married","daughter","son","father","mother","brother","sister",
    "born","died","birth","death","alma","university","college","school",
    "album","song","film","movie","novel","book","poem","play","newspaper",
    "magazine","journal","award","prize","championship","election","company",
    "organization","museum","library","ship","vessel","station","airport",
    "river","mountain","county","city","province","district","island"
}


def _question_segments(question: str) -> List[str]:
    """把 BrowseComp 风格长问题拆成多个可检索 clue 片段。"""
    q = question or ""
    pieces = []
    pieces.extend(re.findall(r'"([^"]{2,100})"', q))
    pieces.extend(re.findall(r'\(([^)]{2,120})\)', q))
    # relative clauses and punctuation often separate independent clues.
    parts = re.split(
        r'[?;:]|,\s*|\b(?:who|whose|which|that|where|when|while|after|before|during)\b',
        q,
        flags=re.I,
    )
    pieces.extend(p.strip(" .,-") for p in parts)
    out, seen = [], set()
    for p in pieces:
        terms = _content_terms(p, max_terms=8)
        if len(terms) < 2:
            continue
        key = " ".join(terms)
        if key not in seen:
            seen.add(key)
            out.append(p.strip())
    return out[:10]


def _relation_terms(question: str) -> List[str]:
    terms = []
    for tok in _tokens(question):
        nt = _norm_token(tok)
        if nt in _RELATION_HINTS and nt not in terms:
            terms.append(nt)
    return terms[:8]


def _clue_groups(question: str, constraints: str = "") -> List[List[str]]:
    groups, seen = [], set()
    for seg in _question_segments(question):
        terms = [_norm_token(t) for t in _content_terms(seg, max_terms=6)]
        if len(terms) < 2:
            continue
        key = " ".join(terms)
        if key not in seen:
            seen.add(key)
            groups.append(terms)
    for line in (constraints or "").splitlines():
        terms = [_norm_token(t) for t in _content_terms(line, max_terms=6)]
        if len(terms) >= 2:
            key = " ".join(terms)
            if key not in seen:
                seen.add(key)
                groups.append(terms)
    rel = _relation_terms(question)
    if len(rel) >= 2:
        groups.append(rel[:6])
    return groups[:12]


def _exact_phrases(question: str) -> List[str]:
    phrases = []
    phrases.extend(re.findall(r'"([^"]{3,100})"', question or ""))
    phrases.extend(re.findall(r'\(([^)]{3,120})\)', question or ""))
    phrases.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b', question or ""))
    out, seen = [], set()
    for p in phrases:
        p = re.sub(r'\s+', ' ', p).strip().lower()
        if len(p.split()) >= 2 and p not in seen:
            seen.add(p)
            out.append(p)
    return out[:8]


def _browsecomp_queries(question: str, constraints: List[str] = None,
                        facts: List[str] = None, limit: int = 6) -> List[str]:
    """针对多 clue hard-negative 题生成互补短查询。"""
    constraints = constraints or []
    facts = facts or []
    terms = _content_terms(question, max_terms=22)
    anchors = _extract_anchors(question + "\n" + "\n".join(facts[-5:]), max_terms=6)
    rel = _relation_terms(question)
    segments = _question_segments(question)
    candidates = []

    if anchors:
        candidates.append(" ".join((anchors[:3] + rel[:2] + terms[:2])[:6]))
    if rel:
        candidates.append(" ".join((anchors[:2] + rel[:4])[:6]))

    for seg in segments[:5]:
        seg_terms = _content_terms(seg, max_terms=6)
        if seg_terms:
            candidates.append(" ".join((anchors[:1] + seg_terms[:5])[:6]))
            candidates.append(" ".join(seg_terms[:6]))

    for c in constraints[:4]:
        c_terms = _content_terms(c, max_terms=6)
        if c_terms:
            candidates.append(" ".join((anchors[:2] + c_terms[:4])[:6]))

    rare = sorted(terms, key=lambda x: (-len(x), terms.index(x)))[:6]
    candidates.append(" ".join(rare[:6]))
    candidates.append(" ".join(terms[:6]))
    return _clean_queries(candidates, limit=limit)


# ══════════════════════════════════════════════════════════════════════
# Screen Agent
# ══════════════════════════════════════════════════════════════════════

class ScreenAgent:
    SYSTEM = "你是筛选智能体。审阅搜索结果，选择与问题最相关的文档阅读。如果都不相关，输出 NONE。"
    PROMPT = """
审阅上述搜索结果。判断哪些文档值得完整阅读。
最多选择2个与题目相关的文档。优先选择同时覆盖多个题目约束、包含具体人名/日期/作品名/地点的文档。
这是 hard-negative 检索任务：只匹配一个常见词或单个泛泛主题的结果通常是干扰项。优先选择覆盖多个稀有线索的文档，并保持结果多样性。

输出：
Relevant DocIDs: <逗号分隔的docid, 或 NONE>"""

    def __init__(self):
        self.history: List[str] = []

    def run(self, client, model, question: str, results: List[Dict],
            facts_text: str, dead_text: str, qh_text: str, round_num: int) -> Dict:
        ctx = (f"## 问题\n{question}\n\n"
               f"## 已确认事实\n{facts_text or '(暂无)'}\n\n"
               f"## 已排除\n{dead_text or '(无)'}\n\n"
               f"## 查询历史\n{qh_text or '(无)'}")[:20000]
        prompt = f"{ctx}\n\n## 搜索结果\n{self._fmt_results(results)}\n\n{self.PROMPT}"
        msgs = [{"role":"system","content":self.SYSTEM},{"role":"user","content":prompt}]
        raw = _chat(client, model, msgs, max_tok=6000)
        docids = self.parse(raw, results)
        return {"docids": docids, "raw": raw}

    def mark_read(self, docids: List[str]):
        for d in docids:
            if d not in self.history:
                self.history.append(d)

    def parse(self, text: str, results: List[Dict]) -> List[str]:
        m = re.search(r'Relevant DocIDs:\s*', text, re.I)
        if not m: return []
        rest = text[m.end():]
        stop = ["Status:","Next Query:","Reasoning:","Key Facts:","Explanation:","Confidence:",
                "Thought:","Action:","Facts Found:","Dead Ends:","Constraint Audit:","Exact Answer:"]
        c = []
        for line in rest.split("\n"):
            s = line.strip()
            if not s: break
            if any(s.startswith(p) for p in stop): break
            c.append(s)
        txt = " ".join(c).strip()
        if not txt or txt.upper()=="NONE": return []
        result_docids = {str(d['docid']): d['docid'] for d in results}
        mapped, seen = [], set()
        for d in re.findall(r'\b(\d+)\b', txt):
            if d in result_docids: actual = result_docids[d]
            elif d.isdigit() and int(d) == 0 and results: actual = results[0]['docid']
            elif d.isdigit() and 1<=int(d)<=len(results): actual = results[int(d)-1]['docid']
            else: continue
            if actual not in seen: seen.add(actual); mapped.append(actual)
        return mapped[:3]

    @staticmethod
    def _fmt_results(results: List[Dict]) -> str:
        L = []
        for i,d in enumerate(results, 1):
            s = d.get('snippet',d.get('text',''))[:3000]
            score = d.get("_score", d.get("score", 0))
            L.append(f"Result {i} | DocID: {d['docid']} | Score: {score:.3f}\n  {s}")
        return "\n\n".join(L) if L else "(无结果)"


# ══════════════════════════════════════════════════════════════════════
# Executor Agent
# ══════════════════════════════════════════════════════════════════════

class ExecutorAgent:
    SYSTEM = "你是提取智能体。从文档中提取与问题相关的具体可验证事实。只报告文档中直接陈述的内容。"
    PROMPT = """
根据上述问题和完整文档文本，分类你的发现：

### 已找到的事实
从文档中提取的与问题相关的具体、可验证的事实。每个事实必须包含 docid，并引用具体细节（名称、日期、标题、事件）。如果没有相关内容，写"None."

### 已排除线索
如果文档看似相关但实际不满足题目约束，说明排除原因；没有则写"None."

输出：
Facts Found:
- 事实1
- 事实2

Dead Ends:
- 排除原因1
"""

    def __init__(self):
        self.facts: List[str] = []
        self.dead_ends: List[str] = []
        self._fact_set: set = set()

    def run(self, client, model, question: str, docs: List[Dict],
            facts_text: str, dead_text: str, qh_text: str) -> Dict:
        if not docs: return {"facts": [], "dead_ends": [], "raw": ""}
        ctx = (f"## 问题\n{question}\n\n"
               f"## 已确认事实\n{facts_text or '(暂无)'}\n\n"
               f"## 已排除\n{dead_text or '(无)'}\n\n"
               f"## 查询历史\n{qh_text or '(无)'}")[:20000]
        prompt = f"{ctx}\n\n## 完整文档\n{self._fmt_docs(docs)}\n\n{self.PROMPT}"
        msgs = [{"role":"system","content":self.SYSTEM},{"role":"user","content":prompt}]
        raw = _chat(client, model, msgs, max_tok=6000)
        new_facts = []
        for f in self._parse_list(raw, "Facts Found"):
            k = f.strip().lower()
            if k and k not in self._fact_set:
                self._fact_set.add(k); self.facts.append(f); new_facts.append(f)
        new_dead = []
        for d in self._parse_list(raw, "Dead Ends"):
            if d and d not in self.dead_ends:
                self.dead_ends.append(d); new_dead.append(d)
        return {"facts": new_facts, "dead_ends": new_dead, "raw": raw}

    def summary(self) -> str:
        return "\n".join(f"- {f}" for f in self.facts[-15:]) if self.facts else "(暂无事实)"

    def dead_summary(self) -> str:
        return "\n".join(f"- {d}" for d in self.dead_ends) if self.dead_ends else "(无)"

    @staticmethod
    def _parse_list(text: str, heading: str) -> List[str]:
        items, ib = [], False
        pat = re.compile(re.escape(heading), re.I)
        for line in text.split("\n"):
            s = line.strip()
            if pat.search(s): ib = True; continue
            if ib and s.startswith("-"):
                item = s[1:].strip()
                if item.lower() in ("none","none.","(none)","...",""): continue
                items.append(item)
            elif ib and s and not s.startswith("-"): break
        return items

    @staticmethod
    def _fmt_docs(docs: List[Dict], mc: int=12000) -> str:
        L = []
        for i,d in enumerate(docs,1):
            t = d.get("text",d.get("error",""))
            L.append(f"--- Document {i} (docid={d['docid']}) ---\n{t[:mc] if len(t)>mc else t}")
        return "\n\n".join(L) if L else "(无文档)"


# ══════════════════════════════════════════════════════════════════════
# Assessor Agent
# ══════════════════════════════════════════════════════════════════════

class AssessorAgent:
    SYSTEM = "你是审计智能体。基于已发现事实链式推进搜索，将已知实体写入查询，判断是否需要继续，并生成高质量BM25查询。"
    PROMPT = """
分析已确认事实和约束缺口，决定下一步。

规则：
- 第一步：先检查已有事实是否足以回答问题。如果能回答，直接输出 ANSWER，不要继续搜索
- 只有当已有事实确实不足以回答时，才输出 SEARCH，针对缺失约束给出2-3个英文BM25查询
- 这是 hard-negative 题：下一轮查询必须补一个具体 clue，不要只扩大主题范围
- 宁可保守判断 ANSWER（事实够了就停），也不要过度搜索导致遗忘早期关键信息
- 链式推进：如果已有事实中包含具体实体（人名、书名、地名、机构名、年份），必须将这些实体写入下一轮查询中，用已知实体去检索缺失的约束，而不是用原问题的泛化词汇重复搜索

BM25查询写作规则（关键）：
3-6个英文词高度相关的能体现题目问题特点的词汇，记得用已知的关键信息来查询

输出：
Status: ANSWER | SEARCH

-- 如果 SEARCH:
Missing Constraint: <本查询试图补足的约束>
Search Query: <英文查询1>
Expected Evidence: <希望命中的文档原文会包含什么>
Missing Constraint: <本查询试图补足的约束>
Search Query: <英文查询2>
Expected Evidence: <希望命中的文档原文会包含什么>
Missing Constraint: <本查询试图补足的约束>
Search Query: <英文查询3>"""

    RETHINK_PROMPT = """
多轮搜索无结果。找全新方向，用完全不同的英文关键词。

输出：
Search Query: <英文查询1>
Search Query: <英文查询2>"""

    def __init__(self):
        self.qh: List[str] = []

    def run(self, client, model, question: str, current_query: str,
            facts_text: str, dead_text: str, constraint_text: str = "") -> Dict:
        self.qh.append(current_query)
        qh_text = "\n".join(f"  [{i+1}] {q}" for i,q in enumerate(self.qh)) if self.qh else "(无)"
        extra = f"最近查询: \"{current_query}\"\n已搜索轮次: {len(self.qh)}"
        ctx = (f"## 问题\n{question}\n\n## 已确认事实\n{facts_text or '(暂无)'}\n\n"
               f"## 已排除\n{dead_text or '(无)'}\n\n## 查询历史\n{qh_text}")
        if constraint_text:
            ctx += f"\n\n## 约束覆盖\n{constraint_text}"
        ctx = f"{ctx}\n\n{extra}"[:20000]
        prompt = f"{ctx}\n\n{self.PROMPT}"
        msgs = [{"role":"system","content":self.SYSTEM},{"role":"user","content":prompt}]
        raw = _chat(client, model, msgs, max_tok=6000)
        return {
            "status": self._parse_status(raw),
            "queries": self._parse_queries(raw),
            "raw": raw,
        }

    def rethink(self, client, model, question: str) -> List[str]:
        """返回重新思考后的新查询列表。"""
        raw = _chat(client, model, [{"role":"system","content":self.SYSTEM},
            {"role":"user","content":f"## 问题\n{question}\n\n{self.RETHINK_PROMPT}"}], 6000)
        return self._parse_queries(raw)

    def fallback_query(self, question: str) -> str:
        """从问题中提取正则关键词。"""
        candidates = []
        candidates.extend(re.findall(r'"([^"]+)"', question))
        candidates.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question))
        candidates.extend(re.findall(r'\b\d{3,4}\b', question))
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
        for i in range(len(words)-1):
            w1,w2 = words[i].lower(),words[i+1].lower()
            if w1 not in _STOPWORDS or w2 not in _STOPWORDS:
                candidates.append(f"{words[i]} {words[i+1]}")
        seen, out = set(), []
        for c in candidates:
            cl = _clean_query(c)
            if cl and cl not in seen:
                seen.add(cl)
                if cl not in {_clean_query(q) for q in self.qh[-3:]}:
                    out.append(cl)
                if len(out)>=5: break
        return _clean_query(" ".join(out[:3])) if out else _clean_query(question)

    def was_recent(self, q: str, w: int = 2) -> bool:
        ql = q.strip().lower()
        return any(ql == h.lower() for h in self.qh[-w:])

    def qh_text(self) -> str:
        return "\n".join(f"  [{i+1}] {q}" for i,q in enumerate(self.qh)) if self.qh else "(无)"

    def count(self) -> int: return len(self.qh)

    @staticmethod
    def _parse_status(text: str) -> str:
        m = re.search(r'Status:\s*\*{0,2}\s*(ANSWER|SEARCH)\s*\*{0,2}', text, re.I)
        return m.group(1).upper() if m else "SEARCH"

    @staticmethod
    def _parse_queries(text: str) -> List[str]:
        """解析多条 Search Query 行。"""
        queries = []
        for m in re.finditer(r'Search Query:\s*"?(.+?)"?\s*$', text, re.I | re.M):
            q = _clean_query(m.group(1).strip().strip('"\''))
            if q and '...' not in q and '<' not in q:
                queries.append(q)
        return _clean_queries(queries, limit=5)


# ══════════════════════════════════════════════════════════════════════
# Synthesizer Agent
# ══════════════════════════════════════════════════════════════════════

class SynthesizerAgent:
    SYSTEM = "你是综合智能体。仅基于已确认事实回答问题。"
    PROMPT = """
所有搜索轮次已完成。根据研究过程中收集的所有已确认事实，精确回答问题。
结合上述列出的事实作为证据回答题目。请注意，回答的答案尽可能完整，不要过度缩写，不丢失语义，必须给出一个最终答案。

输出：
Explanation: <解释原因>
Exact Answer: <精确答案>"""

    def run(self, client, model, question: str, facts_text: str) -> str:
        ctx = f"## 问题\n{question}\n\n## 已确认事实\n{facts_text or '(暂无)'}\n\n{self.PROMPT}"
        msgs = [{"role":"system","content":self.SYSTEM},{"role":"user","content":ctx}]
        return _chat(client, model, msgs, max_tok=6000)


# ══════════════════════════════════════════════════════════════════════
# 约束追踪
# ══════════════════════════════════════════════════════════════════════

class ConstraintTracker:
    """追踪每个约束的覆盖状态。初始化阶段提取约束，每轮更新。"""

    def __init__(self):
        self.constraints: List[Dict] = []  # {id, text, status: verified|partial|missing}

    def extract(self, client, model, question: str):
        """从问题中提取原子约束。"""
        prompt = f"""问题: {question}

提取所有约束条件，每条约束必须原子化(不可再分)。输出JSON:
[{{"id":"C1","text":"born in France"}},{{"id":"C2","text":"won Nobel Prize in 1920s"}}]"""
        msgs = [{"role":"system","content":"从问题提取原子约束。输出JSON数组。"},
                {"role":"user","content":prompt}]
        raw = _chat(client, model, msgs, max_tok=6000)
        data = _parse_json(raw) if callable(_parse_json) else []
        if isinstance(data, list) and data:
            self.constraints = [{"id": c.get("id", f"C{i+1}"), "text": c.get("text",""),
                                 "status": "missing"}
                                for i, c in enumerate(data)]
        elif isinstance(data, dict) and "constraints" in data:
            self.constraints = [{"id": c.get("id", f"C{i+1}"), "text": c.get("text",""),
                                 "status": "missing"}
                                for i, c in enumerate(data["constraints"])]

    def update(self, facts: List[str]):
        """根据已累积事实更新约束状态。"""
        if not self.constraints or not facts:
            return
        for c in self.constraints:
            keywords = [_norm_token(k) for k in _content_terms(c["text"], max_terms=8)]
            if not keywords:
                continue
            best = "missing"
            for fact in facts:
                fact_terms = set(_norm_tokens(fact))
                matches = sum(1 for kw in keywords if kw in fact_terms)
                ratio = matches / max(len(keywords), 1)
                fact_norm = " ".join(_norm_tokens(fact))
                phrase = " ".join(keywords)
                if len(keywords) >= 2 and (phrase in fact_norm or (matches >= 2 and ratio >= 0.8)):
                    best = "verified"
                    break
                if matches > 0:
                    best = "partial"
            if best == "verified" or (best == "partial" and c["status"] == "missing"):
                c["status"] = best

    def missing_summary(self) -> str:
        """返回缺失约束的摘要文本。"""
        if not self.constraints:
            return ""
        verified = [c for c in self.constraints if c["status"] == "verified"]
        partial = [c for c in self.constraints if c["status"] == "partial"]
        missing = [c for c in self.constraints if c["status"] == "missing"]
        parts = []
        if verified:
            parts.append(f"已验证({len(verified)}): " + ", ".join(c["id"] for c in verified))
        if partial:
            parts.append(f"部分({len(partial)}): " + "; ".join(f"{c['id']}:{c['text']}" for c in partial))
        if missing:
            parts.append(f"缺失({len(missing)}): " + "; ".join(f"{c['id']}:{c['text']}" for c in missing))
        return "\n".join(parts)

    def open_constraints(self) -> List[str]:
        """返回仍需要继续检索支撑的约束文本。"""
        return [c["text"] for c in self.constraints if c.get("status") != "verified" and c.get("text")]

    def coverage_score(self) -> float:
        """0-1 分，表示约束覆盖比例。partial=0.5, verified=1.0。"""
        if not self.constraints: return 0.0
        total = len(self.constraints)
        score = sum(1.0 if c["status"] == "verified" else
                    0.5 if c["status"] == "partial" else 0.0
                    for c in self.constraints)
        return score / total

    def all_verified(self) -> bool:
        return bool(self.constraints) and all(c["status"] == "verified" for c in self.constraints)

    def verified_count(self) -> int:
        return sum(1 for c in self.constraints if c["status"] == "verified")

    def best_candidate(self, facts: List[str]) -> str:
        """从事实中找到约束覆盖最多的实体名。"""
        names = set()
        for f in facts:
            for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', f):
                names.add(m.group(1))
            for m in re.finditer(r'\b([A-Z][a-z]+)\b', f):
                name = m.group(1)
                if len(name) > 3 and name[0].isupper():
                    names.add(name)
        if not names: return ""
        best, best_s = "", -1
        facts_text = " ".join(facts).lower()
        for name in names:
            score = 0
            for c in self.constraints:
                kw = re.findall(r'\b[a-zA-Z]{3,}\b', c["text"].lower())
                if any(k in facts_text for k in kw):
                    score += 1
            if score > best_s: best_s = score; best = name
        return best


def _parse_json(text: str):
    """鲁棒 JSON 解析。"""
    t = _strip_think(text)
    t = re.sub(r'```(?:json)?\s*\n?', '', t)
    t = re.sub(r'\n?```', '', t)
    try: return json.loads(t)
    except:
        for pat in [r'\[.*\]', r'\{.*\}']:
            m = re.search(pat, t, re.DOTALL)
            if m:
                try: return json.loads(m.group(0))
                except: pass
    return {}


# ══════════════════════════════════════════════════════════════════════
# 搜索优化
# ══════════════════════════════════════════════════════════════════════

class ReRanker:
    """BM25 top-N → 查询/约束重叠 + BM25 分数重排。"""
    @staticmethod
    def rerank(query: str, results: List[Dict], top_k=7, alpha=0.55,
               question: str = "", constraints: str = "",
               read_docids: set = None):
        if not results:
            return []
        read_docids = read_docids or set()
        scores = [r.get("score", 0.0) for r in results]
        mn, mx = min(scores), max(scores)
        norm = [(s-mn)/(mx-mn) if mx>mn else 0.5 for s in scores]
        qs = set(_norm_token(t) for t in _content_terms(query, max_terms=10)) or set(_norm_tokens(query))
        question_terms = set(_norm_token(t) for t in _content_terms(question, max_terms=16))
        constraint_terms = set(_norm_token(t) for t in _content_terms(constraints, max_terms=20))
        anchors = [_norm_token(a) for a in _extract_anchors(question, max_terms=6)]
        clue_groups = _clue_groups(question, constraints)
        phrases = _exact_phrases(question)
        for i, r in enumerate(results):
            text = " ".join(str(r.get(k, "")) for k in ("title", "snippet", "text"))
            low_text = text.lower()
            ds = set(_norm_tokens(text))
            q_overlap = len(qs & ds) / max(len(qs), 1)
            question_overlap = len(question_terms & ds) / max(len(question_terms), 1) if question_terms else 0
            constraint_overlap = len(constraint_terms & ds) / max(len(constraint_terms), 1) if constraint_terms else 0
            anchor_bonus = 0.06 if any(a in ds for a in anchors) else 0.0
            phrase_bonus = 0.08 if any(p in low_text for p in phrases) else 0.0
            clue_hits = 0
            for group in clue_groups:
                hit = sum(1 for t in group if t in ds)
                if hit >= max(1, min(2, len(group))):
                    clue_hits += 1
            clue_coverage = clue_hits / max(len(clue_groups), 1) if clue_groups else 0.0
            multi_clue_bonus = 0.06 if clue_hits >= 2 else 0.0
            fresh_bonus = 0.04 if r.get("docid") not in read_docids else -0.08
            r["_score"] = (
                alpha * norm[i]
                + 0.24 * q_overlap
                + 0.08 * question_overlap
                + 0.07 * constraint_overlap
                + 0.10 * clue_coverage
                + anchor_bonus
                + phrase_bonus
                + multi_clue_bonus
                + fresh_bonus
            )
        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_k]


def _search(registry: Dict, query: str, question: str = "",
            constraint_text: str = "", read_docids: set = None,
            rerank: bool = True) -> List[Dict]:
    """两步检索：BM25 → ReRanker → top-k。rerank=False 时直接用 BM25 top-5。"""
    try:
        raw = registry["search"](query)
    except: return []
    if not rerank:
        return raw[:5]
    return ReRanker.rerank(query, raw, top_k=5, question=question,
                           constraints=constraint_text, read_docids=read_docids)


def _plan_queries(question: str) -> List[str]:
    """从问题中提取多组关键词查询。"""
    bc_qs = _browsecomp_queries(question, limit=5)
    terms = _content_terms(question, max_terms=18)
    anchors = _extract_anchors(question, max_terms=6)
    rare = sorted(terms, key=lambda x: (-len(x), terms.index(x)))[:6]
    mid = len(terms) // 3
    candidates = [
        " ".join((anchors[:3] + terms[:3])[:6]),
        " ".join(terms[:6]),
        " ".join(terms[mid:mid+6]),
        " ".join((anchors[:2] + rare[:4])[:6]),
    ]
    return _clean_queries(bc_qs + candidates, limit=5)


def _plan_query(question: str) -> str:
    """从问题中提取最独特的关键词（英文），避免使用完整问题当 BM25 查询。"""
    qs = _plan_queries(question)
    return qs[0] if qs else _clean_query(question)


def _constraint_queries(question: str, tracker: ConstraintTracker,
                        facts: List[str], limit: int = 5) -> List[str]:
    """围绕未验证约束生成补缺查询，不改变底层 BM25。"""
    gaps = tracker.open_constraints() if tracker and tracker.constraints else []
    if not gaps:
        return []
    bc_qs = _browsecomp_queries(question, constraints=gaps, facts=facts, limit=limit)
    anchor_text = question + "\n" + "\n".join(facts[-8:])
    anchors = _extract_anchors(anchor_text, max_terms=5)
    candidates = []
    for gap in gaps[:4]:
        gap_terms = _content_terms(gap, max_terms=6)
        if not gap_terms:
            continue
        candidates.append(" ".join((anchors[:2] + gap_terms[:4])[:6]))
        candidates.append(" ".join(gap_terms[:6]))
    return _clean_queries(bc_qs + candidates, limit=limit)


# ══════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════

def run_agent_loop(
    client, model, query: str,
    tools: List[Dict], registry: Dict[str, Any],
    max_turns: int = 10, max_history_msgs: int = 6,
) -> Tuple[str, List[Dict]]:

    print(f"[start] {query[:80]}...", flush=True)

    screen = ScreenAgent()
    executor = ExecutorAgent()
    assessor = AssessorAgent()
    synthesizer = SynthesizerAgent()
    tracker = ConstraintTracker()

    # 初始化：提取约束
    tracker.extract(client, model, query)
    recs = [{"phase":"constraints","constraints":tracker.constraints}]

    searched: set = set()
    read_docids: set = set()
    fcd: Dict[str,int] = {}
    empty = 0
    next_q = ""

    for rnd in range(1, max_turns+1):
        constraint_info = tracker.missing_summary()
        if rnd == 1:
            qs = [query]
        else:
            base_qs = [q.strip() for q in next_q.split("|") if q.strip()] if next_q else []
            gap_qs = _constraint_queries(query, tracker, executor.facts)
            qs = _clean_queries(base_qs + gap_qs, limit=4)

        if rnd>1 and empty>=2:
            rethink_qs = assessor.rethink(client, model, query)
            qs = _clean_queries(rethink_qs + qs, limit=4)

        if not qs:
            fb = assessor.fallback_query(query)
            qs = [fb] if fb else []

        search_qs = []
        for q in qs:
            key = _query_key(q)
            if key and key not in searched:
                search_qs.append(q)
            if len(search_qs) >= 5:
                break
        if not search_qs:
            fb = assessor.fallback_query(query)
            key = _query_key(fb)
            if fb and key not in searched:
                search_qs = [fb]
            else:
                break

        rec: Dict = {"query": " | ".join(search_qs), "search_queries": search_qs,
                     "per_query_results": []}

        # 搜索（合并多查询结果）
        all_results = []
        for q in search_qs:
            searched.add(_query_key(q))
            res = _search(registry, q, question=query,
                          constraint_text=constraint_info,
                          read_docids=read_docids,
                          rerank=(rnd != 1))
            rec["per_query_results"].append({"query": q, "results": res})
            if res: all_results.extend(res)
        # 去重：同一 docid 保留重排分最高的一次
        by_docid = {}
        for r in all_results:
            did = r["docid"]
            old = by_docid.get(did)
            if old is None or r.get("_score", r.get("score", 0)) > old.get("_score", old.get("score", 0)):
                by_docid[did] = r
        results = sorted(by_docid.values(), key=lambda x: x.get("_score", x.get("score", 0)), reverse=True)[:5]
        rec["results"] = results
        if not results: empty+=1; recs.append(rec); continue

        # ScreenAgent 筛选
        scr_msg = screen.run(client, model, query, results,
                             executor.summary(), executor.dead_summary(),
                             assessor.qh_text(), rnd)
        scr_raw = _strip_think(scr_msg["raw"])
        rec["screen"] = scr_raw
        rec["screen_system"] = screen.SYSTEM
        docids = scr_msg["docids"]
        if not docids and results:
            docids = [d["docid"] for d in results[:3]]

        # 获取文档全文
        fresh = [d for d in docids if d not in read_docids]
        docs = _fetch(registry, fresh, fcd) if fresh else []
        read_docids.update(fresh)
        screen.mark_read(fresh)
        rec["fetched"] = docs

        # ExecutorAgent 提取
        ext_msg = executor.run(client, model, query, docs,
                               executor.summary(), executor.dead_summary(),
                               assessor.qh_text())
        ext_raw = _strip_think(ext_msg["raw"]) if ext_msg["raw"] else ""
        rec["extract"] = ext_raw
        rec["executor_system"] = executor.SYSTEM
        empty = 0 if ext_msg["facts"] else empty+1
        if docs and not ext_msg["facts"] and not ext_msg["dead_ends"]:
            tried = ", ".join(str(d.get("docid")) for d in docs)
            note = f"Query '{rec['query']}' fetched docids [{tried}] but yielded no relevant facts."
            if note not in executor.dead_ends:
                executor.dead_ends.append(note)
            rec["auto_dead_end"] = note

        # 更新约束追踪
        tracker.update(executor.facts)

        # AssessorAgent 评估（传入约束覆盖状态）
        constraint_info = tracker.missing_summary()
        ast_msg = assessor.run(client, model, query, rec["query"],
                               executor.summary(), executor.dead_summary(),
                               constraint_info)
        # 把约束信息加到 assessor 的上下文中——修改 assessor.run 的 context 参数
        ast_raw = _strip_think(ast_msg["raw"])
        rec["assess"] = ast_raw
        rec["assessor_system"] = assessor.SYSTEM
        rec["constraint_coverage"] = constraint_info
        recs.append(rec)

        if executor.facts:
            if tracker.all_verified():
                break
            if ast_msg["status"] == "ANSWER" and tracker.all_verified():
                break

        if empty >= 3:
            break

        # 取 Assessor 给出的多条查询，合并为下一轮查询
        nqs = ast_msg["queries"]
        if nqs:
            next_q = " | ".join(_clean_queries(nqs, limit=3))  # 合并多条查询
        else:
            fb = assessor.fallback_query(query)
            if fb: next_q = fb
            else: break

    # Synthesizer：候选名只作为核对线索，不能绕过事实直接返回。
    final_facts = executor.summary()
    constraint_info = tracker.missing_summary()
    if constraint_info:
        final_facts += f"\n\n## 约束覆盖状态\n{constraint_info}"
    best = tracker.best_candidate(executor.facts)
    if best:
        final_facts += f"\n\n## 候选答案线索\n- {best}（必须由上述事实支撑，否则不要直接采用）"
    fa = _strip_think(synthesizer.run(client, model, query, final_facts))
    m = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', fa, re.I)
    if m: fa = m.group(1).strip()

    return fa, _traj(query, recs, fa, synthesizer.SYSTEM)


# ══════════════════════════════════════════════════════════════════════
# 模块级工具
# ══════════════════════════════════════════════════════════════════════

def _fetch(registry: Dict, docids: List[str], fcd: Dict[str,int]) -> List[Dict]:
    gf = registry["get_document"]; docs = []
    for did in docids[:3]:
        if fcd.get(did,0)>=2: continue
        try: doc = gf(did)
        except: doc = None
        if doc is None or (isinstance(doc,dict) and "error" in doc):
            docs.append({"docid":did,"error":"not found"})
        else: fcd[did]=fcd.get(did,0)+1; docs.append({"docid":did,"text":doc.get("text",""),"url":doc.get("url","")})
    return docs


def _traj(question, recs, final_answer, synthesizer_sys=""):
    msgs = [{"role":"system","content":"多智能体系统 — Screen/Executor/Assessor/Synthesizer"},
            {"role":"user","content":question}]
    cid = 0
    for rec in recs:
        if rec.get("phase") == "constraints": continue  # skip constraint phase record
        if "query" not in rec: continue
        per_query = rec.get("per_query_results") or [{"query": rec["query"], "results": rec.get("results", [])}]
        for qr in per_query:
            cid+=1
            msgs.append({"role":"assistant","content":"",
                "tool_calls":[{"id":f"call_{cid}","type":"function",
                    "function":{"name":"search","arguments":json.dumps({"query":qr.get("query","")},ensure_ascii=False)}}]})
            msgs.append({"role":"tool","tool_call_id":f"call_{cid}",
                "content":json.dumps([{"docid":d["docid"],"score":float(d.get("score",0)),
                    "rerank_score":round(float(d.get("_score", d.get("score",0))), 4)}
                    for d in qr.get("results",[])],ensure_ascii=False)})
        if rec.get("screen_system"):
            msgs.append({"role":"assistant","content":f"[Screen Agent · {rec['screen_system']}]\n{rec.get('screen','')}"})
        else:
            msgs.append({"role":"assistant","content":f"[Screen Agent]\n{rec.get('screen','')}"})
        for d in (rec.get("fetched") or []):
            cid+=1
            msgs.append({"role":"assistant","content":"",
                "tool_calls":[{"id":f"call_{cid}","type":"function",
                    "function":{"name":"get_document","arguments":json.dumps({"docid":d["docid"]},ensure_ascii=False)}}]})
            msgs.append({"role":"tool","tool_call_id":f"call_{cid}",
                "content":json.dumps({"docid":d["docid"],
                    "text_preview":(d.get("text","") or "")[:500],"url":d.get("url","")},ensure_ascii=False)})
        executor_text = rec.get('extract','')
        if rec.get("auto_dead_end"):
            executor_text = (executor_text + "\n" if executor_text else "") + f"Dead Ends:\n- {rec['auto_dead_end']}"
        if rec.get("executor_system"):
            msgs.append({"role":"assistant","content":f"[Executor Agent · {rec['executor_system']}]\n{executor_text}"})
        else:
            msgs.append({"role":"assistant","content":f"[Executor Agent]\n{executor_text}"})
        if rec.get("assessor_system"):
            msgs.append({"role":"assistant","content":f"[Assessor Agent · {rec['assessor_system']}]\n{rec.get('assess','')}"})
        else:
            msgs.append({"role":"assistant","content":f"[Assessor Agent]\n{rec.get('assess','')}"})
    if synthesizer_sys:
        msgs.append({"role":"assistant","content":f"[Synthesizer Agent · {synthesizer_sys}]\n{final_answer}"})
    else:
        msgs.append({"role":"assistant","content":f"[Synthesizer Agent]\n{final_answer}"})
    return msgs
