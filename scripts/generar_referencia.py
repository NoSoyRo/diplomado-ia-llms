#!/usr/bin/env python3
"""Regenera `parte-3-oficio/salidas_referencia.json`: salidas reales de Qwen sobre el hold-out.

Dos condiciones, mismo modelo, greedy:

* zero-shot: system + notas.
* few-shot:  system + 1 ejemplo resuelto del dataset (user/assistant) + notas.

* lora:      system + notas, con el adapter de `--adapter` (si se pasa).

    python scripts/generar_referencia.py
    python scripts/generar_referencia.py --adapter parte-3-oficio/don-titular-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parents[1]
OFICIO = RAIZ / "parte-3-oficio"
sys.path.insert(0, str(OFICIO))
import don_titular as dt  # noqa: E402

MODELO = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 220


def prompts(notas: str, ejemplo: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        "zero-shot": dt.mensajes(notas),
        "few-shot": [ejemplo[0], ejemplo[1], ejemplo[2], {"role": "user", "content": notas}],
    }


def generar(model, tok, msgs: list[dict[str, str]]) -> str:
    entrada = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    entrada = entrada.to(model.device)
    with torch.no_grad():
        out = model.generate(
            **entrada,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, entrada["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="", help="carpeta del adapter LoRA (agrega la columna lora)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODELO)
    model = AutoModelForCausalLM.from_pretrained(MODELO).eval()
    con_lentes = None
    if args.adapter:
        from peft import PeftModel

        con_lentes = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODELO), args.adapter).eval()
    ejemplo = json.loads((OFICIO / "ejemplo-briefing.jsonl").read_text(encoding="utf-8").splitlines()[0])["messages"]
    holdout = [json.loads(linea) for linea in (OFICIO / "holdout.jsonl").read_text(encoding="utf-8").splitlines()]

    salidas: dict[str, dict[str, str]] = {}
    for caso in holdout:
        salidas[caso["id"]] = {}
        condiciones = prompts(caso["notas"], ejemplo)
        for nombre, msgs in condiciones.items():
            salidas[caso["id"]][nombre] = generar(model, tok, msgs)
        if con_lentes is not None:
            salidas[caso["id"]]["lora"] = generar(con_lentes, tok, condiciones["zero-shot"])
        for nombre, texto in salidas[caso["id"]].items():
            print(f"── {caso['id']} · {nombre}\n{texto}\n→ {dt.juzgar(caso['notas'], texto).resumen()}\n")

    meta = {"modelo": MODELO, "decodificacion": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "fecha": date.today().isoformat()}
    if con_lentes is not None:
        cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text(encoding="utf-8"))
        entreno = json.loads((Path(args.adapter) / "entrenamiento.json").read_text(encoding="utf-8")) if (Path(args.adapter) / "entrenamiento.json").exists() else {}
        meta["lora"] = {"r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "target_modules": sorted(cfg["target_modules"]), **entreno}
    destino = OFICIO / "salidas_referencia.json"
    destino.write_text(json.dumps({"meta": meta, "salidas": salidas}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"escrito {destino}")


if __name__ == "__main__":
    main()
