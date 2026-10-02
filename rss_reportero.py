#!/usr/bin/env python3
"""Reportero del noticiero — cero IA.

Dummy
-----
Un LLM no "sabe el día". Si le preguntas qué pasó hoy sin notas, inventa
(el villano del módulo). Este script es el reportero: baja un RSS público
y arma el bloque de notas que luego pegas al modelo. Cero inteligencia
artificial. Si esto falla, el noticiero falla — y no es culpa de Qwen.

Arquitectura del producto (no se borra del pizarrón)::

    [RSS / API]  →  paquete de notas de HOY  →  LLM (+ LoRA)  →  briefing
     reportero            el papel                  conductor

¿Dónde está "el día"? En el XML. ¿Dónde está "el estilo"? En el LLM.

Math (RAG casero)
-----------------
Un retriever R (aquí: parsear el feed) produce documentos d = R(q).
Se genera

    ŷ ~ P_θ( · | φ(system, q, d) )

es decir P(y | q, d) en vez de P(y | q). φ es el chat template. No hace
falta un vector database el día 1: un string d basta. Si el feed viene
vacío, d está vacío y no hay noticiero. Eso es un fallo observable.

`--n` es el máximo de notas que *sobreviven* al filtro `--query`, no las
primeras n del XML. Si filtraras después de recortar, `--query economia`
podría devolver vacío aunque más abajo hubiera economía.

Uso
---
    python rss_reportero.py
    python rss_reportero.py --feed https://feeds.bbci.co.uk/mundo/rss.xml --n 8
    python rss_reportero.py --query economia
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date

DEFAULT_FEED = "https://feeds.bbci.co.uk/mundo/rss.xml"


def strip_tags(raw: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def fetch_feed(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "llms-diplomado-reportero/1.0 (uso educativo)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def parse_items(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    items: list[dict[str, str]] = []
    for item in root.iter("item"):
        title = strip_tags(item.findtext("title") or "")
        desc = strip_tags(item.findtext("description") or "")
        link = (item.findtext("link") or "").strip()
        if title:
            items.append({"title": title, "desc": desc, "link": link})
    return items


def format_notes(items: list[dict[str, str]], query: str | None, limit: int) -> str:
    if query:
        q = query.lower()
        items = [it for it in items if q in (it["title"] + " " + it["desc"]).lower()]
    items = items[: max(1, limit)]
    lines = [f"Notas del {date.today().isoformat()}:"]
    for i, it in enumerate(items, 1):
        snippet = it["desc"][:220]
        lines.append(f"{i}. {it['title']}")
        if snippet:
            lines.append(f"   {snippet}")
        if it["link"]:
            lines.append(f"   fuente: {it['link']}")
    if len(lines) == 1:
        lines.append("(el feed no trajo notas para ese filtro)")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Baja un RSS y arma el papel del noticiero.")
    p.add_argument("--feed", default=DEFAULT_FEED, help="URL del RSS")
    p.add_argument("--n", type=int, default=10, help="máximo de notas DESPUÉS del filtro")
    p.add_argument("--query", default="", help="filtro opcional (economía, metro…)")
    p.add_argument("--out", default="", help="si se indica, escribe un .txt")
    args = p.parse_args()

    try:
        raw = fetch_feed(args.feed)
    except OSError as exc:
        print(f"El reportero falló (red): {exc}", file=sys.stderr)
        print("Sin papel no hay noticiero. No inventes el día a mano.", file=sys.stderr)
        return 1

    items = parse_items(raw)
    block = format_notes(items, args.query or None, args.n)
    print(block)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(block + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
