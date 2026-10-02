#!/usr/bin/env python3
"""Regenera `parte-3-oficio/salidas_referencia.json`: salidas reales de Qwen sobre el hold-out.

Dos condiciones, mismo modelo, greedy:

* zero-shot: system + notas.
* few-shot:  system + 1 ejemplo resuelto del dataset (user/assistant) + notas.

La columna LoRA no está aquí: la genera cada quien con su adapter en el notebook 07.

    python scripts/generar_referencia.py
"""

from __future__ import annotations

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


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODELO)
    model = AutoModelForCausalLM.from_pretrained(MODELO).eval()
    ejemplo = json.loads((OFICIO / "ejemplo-briefing.jsonl").read_text(encoding="utf-8").splitlines()[0])["messages"]
    holdout = [json.loads(linea) for linea in (OFICIO / "holdout.jsonl").read_text(encoding="utf-8").splitlines()]

    salidas: dict[str, dict[str, str]] = {}
    for caso in holdout:
        salidas[caso["id"]] = {}
        for nombre, msgs in prompts(caso["notas"], ejemplo).items():
            entrada = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
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
            texto = tok.decode(out[0, entrada["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            salidas[caso["id"]][nombre] = texto
            print(f"── {caso['id']} · {nombre}\n{texto}\n→ {dt.juzgar(caso['notas'], texto).resumen()}\n")

    meta = {"modelo": MODELO, "decodificacion": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "fecha": date.today().isoformat()}
    destino = OFICIO / "salidas_referencia.json"
    destino.write_text(json.dumps({"meta": meta, "salidas": salidas}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"escrito {destino}")


if __name__ == "__main__":
    main()
