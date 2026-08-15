#!/usr/bin/env python3
"""Gym de Don Titular — SFT + LoRA sobre un Instruct chico.

Default del aula: transformers + peft + trl (PyTorch legible).
Unsloth es opcional y solo si tienes NVIDIA.

    pip install "transformers>=4.45" peft trl datasets accelerate
    # GPU 4-bit (Colab T4):
    pip install bitsandbytes

    python sft_lora_noticias.py --data ejemplo-briefing.jsonl

Sustituye el JSONL por tus 200+ ejemplos revisados (regla de oro:
si el assistant menciona algo que no está en el user, se tira).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="ejemplo-briefing.jsonl")
    p.add_argument("--out", default="don-titular-lora")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--4bit", dest="fourbit", action="store_true", help="QLoRA (bitsandbytes)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"No está el JSONL: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"trust_remote_code": True}
    if args.fourbit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    ds = load_dataset("json", data_files=str(data_path), split="train")
    split = ds.train_test_split(test_size=0.1, seed=args.seed)

    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    sft_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=args.seed,
        max_length=args.max_seq,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Adapter en {args.out}/  — esto son los lentes, no el cerebro.")


if __name__ == "__main__":
    main()
