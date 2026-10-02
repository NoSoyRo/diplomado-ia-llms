#!/usr/bin/env python3
"""Gym de Don Titular — SFT + LoRA sobre un Instruct chico.

Dummy
-----
No le haces cirugía al cerebro. Le pones lentes.

Un Qwen 0.5B ya habla español. Tocar sus cientos de millones de pesos:
se te olvida el idioma, no cabe en Colab, tarda una eternidad. LoRA deja
el cerebro W congelado y entrena dos matrices flacas A, B cuyo producto
es un parche de rango r (8 o 16). Se quitan los lentes y el chef vuelve
a ser el de Hugging Face.

Los lentes NO guardan las noticias del 14 de agosto. Guardan el oficio:
5 bullets, cita el medio, di "no aparece". El día sigue viniendo del
reportero (rss_reportero.py).

Math · LoRA (Hu et al., 2021)
-----------------------------
Para una matriz congelada W ∈ R^{d×k} se entrena solo un update de
rango r ≪ min(d, k):

    W' = W + (α / r) B A,    B ∈ R^{d×r},  A ∈ R^{r×k}

Params entrenables: r(d + k) en vez de d·k. Típico: r ∈ {8, 16},
α ~ 2r, aplicado a W_Q y W_V (este script; puedes sumar k_proj / o_proj).
El archivo son {A, B} (decenas de MB). QLoRA (--4bit) guarda W en 4 bits
y descuantiza al vuelo; A, B siguen en bf16. En CPU el truco casi no paga.

Math · SFT enmascarado
----------------------
Mismo next-token del lab 0. Otra máscara. Sea m_t = 1 si el token t es
del assistant y 0 si es system/user:

    L_SFT(θ) = - 1/(Σ m_t)  Σ_t m_t log P_θ(x_t | x_<t)

El enunciado (las notas) no se "practica": es contexto. El modelo
practica a escribir el briefing. TRL aplica esa máscara al formato
`messages` del JSONL. DPO/RLHF reponderan después con y⁺ ≻ y⁻; no
entran a este lab.

Regla de oro
------------
Si el assistant menciona algo que no está en el user, se tira el ejemplo.
Anti-alucinación con datos, no con un sermón. Este JSONL tiene 3 filas
(plantilla). Sustitúyelo por 200+ revisados. Con 3 filas y test_size=0.1
el eval es de juguete: el script avisa.

    pip install "transformers>=4.45" peft trl datasets accelerate
    # GPU 4-bit (Colab T4):
    pip install bitsandbytes

    python sft_lora_noticias.py --data ejemplo-briefing.jsonl
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
    n = len(ds)
    if n < 20:
        print(
            f"Aviso: {n} ejemplos. Esto es la plantilla, no el gym. "
            "Con test_size=0.1 el eval queda en 0–1 fila. Sustituye por 200+ revisados."
        )
    test_size = 0.1 if n >= 10 else 0.0
    if test_size > 0:
        split = ds.train_test_split(test_size=test_size, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    # B_eff = batch micro × acumulación  (mismo truco del ejercicio 6)
    print(f"batch_efectivo = {args.bs} × {args.accum} = {args.bs * args.accum}")

    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    sft_kwargs: dict = {
        "output_dir": args.out,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.bs,
        "gradient_accumulation_steps": args.accum,
        "learning_rate": args.lr,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "seed": args.seed,
        "max_length": args.max_seq,
        "report_to": "none",
    }
    if eval_ds is not None:
        sft_kwargs["per_device_eval_batch_size"] = args.bs
        sft_kwargs["eval_strategy"] = "epoch"
    sft_args = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Adapter en {args.out}/  — esto son los lentes, no el cerebro.")


if __name__ == "__main__":
    main()
