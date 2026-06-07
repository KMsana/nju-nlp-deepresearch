# OpenTrack 多智能体深度研究 Agent

**最终得分**：34% (17/50)

## 方案概述

OpenTrack 将 ReAct 的单模型自主决策解耦为四个专业化 Agent（ScreenAgent / ExecutorAgent / AssessorAgent / SynthesizerAgent），由 Orchestrator 代码硬编码串联。

## 运行方式

### 1. 环境准备

```bash
pip install vllm openai jsonlines
pip install -r ../core/agent/requirements.txt
```

确保 BM25 索引已构建，vLLM 服务已启动（参见主 README）。

### 2. 运行

**推荐方式：Notebook**

打开 `opentrack.ipynb`，按顺序执行单元格即可。

**命令行方式**

```bash
cd open_track
python run.py
```

关键配置（在 `run.py` 中）：
```python
MAX_TURNS = 5
MODEL_NAME = "qwen_auto"
```

### 3. 评估

```bash
python -m core.agent.eval \
  --submission runs/submission.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/eval_results.jsonl
```

## 评估结果

已提交轨迹和评估结果见 `eval/` 目录。

## SFT 微调（未采用）

参见 `train.py`。训练数据 HotpotQA（外部数据集），微调后准确率 24%，低于基座模型，最终未采用。
