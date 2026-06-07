# Deep Research Agent — OpenTrack 多智能体框架

**姓名**：林丞毅　**学号**：231300025　**最终得分**：34% (17/50)

---

## 1. 方案概述

本实验基于 BrowseComp-Plus 数据集搭建多轮检索 agent。实现了三种方案：

| 方案 | 入口 | 准确率 | 说明 |
|---|---|---|---|
| BaseReact | `core/agent/deepresearch_agent.py` | 12% | 纯 ReAct，模型自主决策 |
| ReAct | `open_track/agent/react_loop.py` 提交目录外 | 18% | 状态机 + EvidenceStore + 纠偏 |
| **OpenTrack** | `open_track/agent/multiagent.py` | **34%** | 多 Agent + 多路召回 + ReRanker |

额外尝试了 SFT 微调实验（24%，未采用）。详见 [`open_track/README.md`](open_track/README.md) 和实验报告。

## 2. 文件结构

```
├── 林丞毅-231300025-acc=34-opentrack.pdf   # 实验报告
├── core/
│   ├── deepresearch.ipynb                  # 实验主 Notebook
│   └── agent/                              # 基础模块（检索/评估/客户端）
├── eval/                                   # Baseline 评估结果
│   ├── *-submission-12_basereact.jsonl     # BaseReact (12%)
│   ├── *-eval_results-12_basereact.jsonl
│   ├── *-submission-18_react.jsonl         # ReAct (18%)
│   └── *-eval_results-18_react.jsonl
├── open_track/                             # OpenTrack 额外提交
│   ├── README.md                           # OpenTrack 独立说明
│   ├── opentrack.ipynb                     # OpenTrack Notebook 入口
│   ├── run.py                              # OpenTrack 命令行入口
│   ├── agent/                              # OpenTrack 代码
│   │   └── multiagent.py                   #   完整版
│   ├── eval/                               # OpenTrack 评估结果
│   │   ├── *-submission-34.jsonl
│   │   └── *-eval_results-34.jsonl
│   ├── train.py                            # SFT 训练脚本
│   ├── gen_distill_data.py                 # 蒸馏数据生成
│   └── data/                               # HotpotQA 训练数据
└── README.md
```

## 3. 环境准备

### 3.1 模型下载

```bash
git clone https://atomgit.com/hf_mirrors/MindSpore-Lab/Qwen3-8B.git
```

### 3.2 依赖安装

```bash
pip install vllm openai jsonlines tiktoken
pip install -r core/agent/requirements.txt
```

### 3.3 构建 BM25 索引

```bash
# 先将语料库放到 browsecomp-plus-corpus/data/ 下（或软链到实际路径）
mkdir -p indexes
python -m core.agent.build_bm25_index \
  --corpus-path browsecomp-plus-corpus/data/ \
  --index-path indexes/browsecomp_plus_bm25.sqlite
```

## 4. 启动 vLLM 服务

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

服务地址：`http://127.0.0.1:8000/v1`

## 5. 运行 Agent

### Baseline（BaseReact / ReAct）

使用 Notebook：`core/deepresearch.ipynb`

### OpenTrack

使用 Notebook：`open_track/opentrack.ipynb`（与 baseline 运行方式一致）

命令行备选：`cd open_track && python run.py`

详见 [`open_track/README.md`](open_track/README.md)。

## 6. 自动评估

```bash
python -m core.agent.eval \
  --submission runs/submission.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/eval_results.jsonl
```