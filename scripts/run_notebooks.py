#!/usr/bin/env python3
"""Ejecuta notebooks de punta a punta (cada uno desde su carpeta).

Por defecto corre los que no necesitan torch, GPU ni red: 04, 06, 07.
Si una celda truena, el script truena: así CI detecta un notebook roto.

    python scripts/run_notebooks.py                       # los de CI
    python scripts/run_notebooks.py parte-2-chef/05_*.ipynb --save
    python scripts/run_notebooks.py --save                # guarda salidas en el .ipynb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
SIN_TORCH = [
    "parte-2-chef/04_bpe_a_mano.ipynb",
    "parte-3-oficio/06_reportero_y_dataset.ipynb",
    "parte-3-oficio/07_juicio_antes_despues.ipynb",
]


def correr(path: Path, guardar: bool, timeout: int) -> float:
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    if guardar:
        nbformat.write(nb, path)
    return time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("notebooks", nargs="*", help=f"rutas relativas a la raíz (default: {', '.join(SIN_TORCH)})")
    p.add_argument("--save", action="store_true", help="escribe las salidas en el .ipynb")
    p.add_argument("--timeout", type=int, default=900, help="segundos por celda")
    args = p.parse_args()

    fallos = 0
    for rel in args.notebooks or SIN_TORCH:
        path = (RAIZ / rel).resolve()
        try:
            dt = correr(path, args.save, args.timeout)
            print(f"ok    {rel}  ({dt:.0f}s)")
        except Exception as exc:
            fallos += 1
            print(f"FALLA {rel}\n{exc}", file=sys.stderr)
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
