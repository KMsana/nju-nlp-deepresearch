# -*- coding: utf-8 -*-
"""
LoRA SFT 训练 — 读取已有的 train.jsonl / val.jsonl 进行微调

用法:
  python open_track/train.py --model-path /path/to/Qwen3-8B
  python open_track/train.py --model-path ./Qwen3-8B --epochs 5 --lr 1e-4
"""

import argparse, json, sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "nju-nlp-deepresearch"))

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"


def _detect_device():
    try:
        import torch
        import torch_npu
        if torch.npu.is_available():
            return "npu", torch.float16, "npu:0"
    except (ImportError, RuntimeError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16, "cuda"
    except (ImportError, RuntimeError):
        pass
    import torch
    return "cpu", torch.float32, "cpu"


def run_sft(model_path, train_path, val_path, ckpt_dir, merged_dir,
            dev, dtype, devstr, epochs, lr, batch_size, grad_accum, max_samples):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"[train] 设备: {dev}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = [json.loads(l) for l in open(train_path, encoding="utf-8") if l.strip()]
    val_raw = [json.loads(l) for l in open(val_path, encoding="utf-8") if l.strip()]
    if len(train_raw) > max_samples:
        train_raw = train_raw[:max_samples]
    print(f"[train] train {len(train_raw)}, val {len(val_raw)}")

    def _tok(samples):
        out = []
        for s in samples:
            msgs = s["messages"]
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            pids = tokenizer.encode(prompt, add_special_tokens=False)
            fids = tokenizer.encode(full, add_special_tokens=False)
            labels = [-100] * len(pids) + fids[len(pids):]
            if len(fids) > 1024:
                fids = fids[:1024]; labels = labels[:1024]
            pad = 1024 - len(fids)
            out.append({
                "input_ids": torch.tensor(fids + [tokenizer.pad_token_id] * pad, dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(fids) + [0] * pad, dtype=torch.long),
                "labels": torch.tensor(labels + [-100] * pad, dtype=torch.long),
            })
        return out

    print("[train] tokenizing...")
    tr_d, va_d = _tok(train_raw), _tok(val_raw)

    if dev == "npu":
        import torch_npu
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True).to(devstr)
    elif dev == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True)

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    ta = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr, warmup_ratio=0.1,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False, bf16=(dev == "cuda"), report_to="none",
        gradient_checkpointing=True,
        optim="adafactor",
    )

    from transformers import Trainer
    class _T(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            o = model(**inputs)
            sl = o.logits[..., :-1, :].contiguous()
            ll = labels[..., 1:].contiguous()
            loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                sl.view(-1, sl.size(-1)), ll.view(-1))
            return (loss, o) if return_outputs else loss

    class _D(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): return self.d[i]

    trainer = _T(model=model, args=ta,
                 train_dataset=_D(tr_d), eval_dataset=_D(va_d))
    print("[train] 开始训练...")
    trainer.train()

    best = f"{ckpt_dir}_best"
    model.save_pretrained(best); tokenizer.save_pretrained(best)
    print(f"[train] LoRA → {best}")

    # 合并导出
    from peft import PeftModel
    if dev == "npu":
        import torch_npu
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True).to(devstr)
    elif dev == "cuda":
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, best).merge_and_unload()
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir); tokenizer.save_pretrained(merged_dir)
    print(f"[train] 合并 → {merged_dir}")


def main():
    p = argparse.ArgumentParser(description="LoRA SFT 微调")
    p.add_argument("--model-path", required=True, help="基座模型路径")
    p.add_argument("--train-data", default=str(DATA_DIR / "train.jsonl"))
    p.add_argument("--val-data", default=str(DATA_DIR / "val.jsonl"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "checkpoints"))
    p.add_argument("--merged-dir", default=str(OUTPUT_DIR / "merged"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=1000, help="最大训练样本数")
    args = p.parse_args()

    dev, dtype, devstr = _detect_device()

    run_sft(args.model_path, args.train_data, args.val_data,
            args.output_dir, args.merged_dir,
            dev, dtype, devstr,
            args.epochs, args.lr, args.batch_size, args.grad_accum,
            args.max_samples)


if __name__ == "__main__":
    main()
