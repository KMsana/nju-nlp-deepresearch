# -*- coding: utf-8 -*-
"""
蒸馏数据生成 — 只训练 query planning（问题 → BM25 搜索查询）

数据源: HotpotQA
用法:
  python open_track/gen_distill_data.py --api-key sk-xxx --max-questions 2000
"""

import argparse, json, os, re, sys, urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"
HOTPOT_URLS = [
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    "https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpotqa/dataset/dev.json",
]

SYSTEM_PROMPT = "生成 BM25 搜索查询。输出 JSON 数组。"

USER_TEMPLATE = """问题: {question}

为这个问题生成 5 条 BM25 搜索查询，用于在文档语料库中找到回答问题所需的信息。

规则：
- 每条查询 2-6 个英文实词
- 不同查询覆盖问题的不同角度/线索
- 短、精确、有区分度
- 不要使用虚词（the, is, who, what, which...）

输出 JSON 数组：["query1", "query2", ...]"""


_STOPWORDS = {
    'the','a','an','is','was','are','were','be','been','being','in','on','at',
    'to','for','of','from','by','with','and','or','but','not','this','that',
    'these','those','can','you','tell','find','what','when','where','who','how',
    'why','which','name','one','first','last','mid','there','their','they','them',
    'has','have','had','its','also','about','after','before','during','into',
    'over','under','between','among','whose','whom','did','does','do','known',
    'called','would','could','should','answer','question',
}


def _tokens(text):
    return [t.lower().strip("'") for t in re.findall(r"[A-Za-z0-9]+", text or "")]


def _clean_query(q, min_terms=2, max_terms=6):
    q = re.sub(r'(?i)^\s*(search query|query|keywords?)\s*:\s*', '', q or "")
    terms, seen = [], set()
    for tok in _tokens(q):
        if tok in _STOPWORDS or (len(tok) < 3 and not tok.isdigit()):
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
        if len(terms) >= max_terms:
            break
    return " ".join(terms) if len(terms) >= min_terms else ""



def _parse_json_arr(raw):
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'<think>.*$', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'```(?:json)?\s*\n?', '', raw)
    raw = re.sub(r'\n?```', '', raw)
    for pat in [r'\[.*?\]', r'```json\s*\n?(.*?)\n?```']:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1) if m.lastindex else m.group(0))
            except json.JSONDecodeError:
                pass
    return None



def load_hotpot(max_questions):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = DATA_DIR / "hotpot_dev_fullwiki_v1.parquet"
    json_path = DATA_DIR / "hotpot_dev_fullwiki_v1.json"
    if parquet_path.exists():
        print(f"      加载 parquet: {parquet_path}")
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(parquet_path))
            rows = table.to_pylist()
        except ImportError:
            print("      需要 pyarrow: pip install pyarrow")
            sys.exit(1)
        questions = []
        for item in rows:
            q = item.get("question", "")
            if q:
                questions.append(q)
                if len(questions) >= max_questions:
                    return questions
        return questions

    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        questions = []
        for item in data:
            q = item.get("question", "")
            if q:
                questions.append(q)
                if len(questions) >= max_questions:
                    return questions
        return questions

    # 下载
    print(f"[init] 下载 HotpotQA...")
    hf_url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/validation-00000-of-00001.parquet"
    for url in [hf_url] + HOTPOT_URLS:
        try:
            print(f"      尝试 {url[:70]}...")
            urllib.request.urlretrieve(url, parquet_path if url == hf_url else json_path)
            print(f"      下载成功")
            return load_hotpot(max_questions)
        except Exception as e:
            print(f"      失败: {e}")
    print("[error] 下载失败，请手动下载放到 data/ 目录")
    sys.exit(1)


def call_deepseek(base_url, api_key, model, messages, temperature=0.0):
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": 512,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} | {body[:200]}")


def extract_queries(resp_text):
    """提取查询列表，和 multiagent.py 的 _clean_queries 逻辑一致。"""
    arr = _parse_json_arr(resp_text)
    if not arr or not isinstance(arr, list):
        return []
    cleaned, seen = [], set()
    for item in arr:
        cq = _clean_query(str(item))
        if not cq:
            continue
        key = " ".join(_tokens(cq))
        if key not in seen:
            seen.add(key)
            cleaned.append(cq)
        if len(cleaned) >= 5:
            break
    return cleaned


def _process_one(question, base_url, api_key, model):
    """处理单个问题（两次温度采样），返回 (question, queries) 或 None。"""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]
    all_queries, seen = [], set()
    for temp in [0.0, 0.5]:
        try:
            resp = call_deepseek(base_url, api_key, model, msgs, temperature=temp)
            text = resp["choices"][0]["message"]["content"]
            for cq in extract_queries(text):
                if cq not in seen:
                    seen.add(cq)
                    all_queries.append(cq)
        except Exception:
            pass
    return (question, all_queries) if all_queries else None


def generate(api_key, base_url, model, max_questions):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    print(f"[1/3] 加载 HotpotQA...")
    questions = load_hotpot(max_questions)
    print(f"      {len(questions)} 个问题")

    workers = 5
    print(f"[2/3] 并行调用 {model}（{workers} 线程）...")
    samples = []
    done_count = 0
    errors = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, q, base_url, api_key, model): q
                   for q in questions}
        for future in as_completed(futures):
            with lock:
                done_count += 1
            result = future.result()
            if result:
                q, queries = result
                samples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(question=q)},
                        {"role": "assistant", "content": json.dumps(queries[:5], ensure_ascii=False)},
                    ],
                })
            else:
                with lock:
                    errors += 1
            if done_count % 100 == 0:
                print(f"      {done_count}/{len(questions)}, 样本 {len(samples)}, 错误 {errors}")

    print(f"      {done_count}/{len(questions)}, 样本 {len(samples)}, 错误 {errors}")

    print(f"[3/3] 保存...")
    import random
    random.seed(42)
    random.shuffle(samples)
    vs = max(1, int(len(samples) * 0.1))
    val_s, train_s = samples[:vs], samples[vs:]

    for name, data in [("train", train_s), ("val", val_s)]:
        path = str(DATA_DIR / f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"      {name}: {len(data)} 条 -> {path}")

    print(f"\n完成! {len(samples)} 条样本 (来自 {len(questions)} 个问题)")


def main():
    p = argparse.ArgumentParser(description="生成查询蒸馏数据")
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    p.add_argument("--base-url", default="")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--max-questions", type=int, default=2000)
    args = p.parse_args()

    if not args.api_key:
        print("需要 --api-key ")
        sys.exit(1)

    generate(args.api_key, args.base_url, args.model, args.max_questions)


if __name__ == "__main__":
    main()
