#!/usr/bin/env python3
"""Juez de Don Titular — la regla de oro, en código.

Dummy
-----
Un briefing es bueno si (1) tiene la forma pedida (5 bullets numerados),
(2) cada hecho dice de qué medio salió y ese medio está en las notas, y
(3) lo que afirma se puede señalar con el dedo en las notas. Si el modelo
escribe "terremoto en Oaxaca" y las notas no dicen "terremoto", eso es
inventar, aunque suene a noticia.

Este archivo no usa IA. Es el mismo juez para dos momentos:

* **Antes de entrenar**: `validar_ejemplo(fila)` decide si una fila del
  JSONL se queda o se tira. El modelo copia lo que ve.
* **Después de entrenar**: `juzgar(notas, briefing)` califica una salida
  del modelo (tabla antes vs después).

Math
----
Sea d el texto de las notas y b un bullet. Con S(·) = conjunto de raíces
(primeras 5 letras sin acento) de palabras de contenido y N(·) = conjunto
de números:

    soporte(b, d) = |S(b) ∩ S(d)| / |S(b)|

b está soportado si soporte ≥ τ (τ = 0.5) y N(b) ⊆ N(d). Un número que no
está en las notas casi siempre es inventado; por eso es duro, no blando.

Es una cota barata, no un oráculo: un bullet puede reusar las palabras de
las notas y aun así decir lo contrario ("Banxico *sube* la tasa"). El juez
humano sigue existiendo; este solo filtra lo obvio a escala.

Uso
---
    python don_titular.py ejemplo-briefing.jsonl     # valida un dataset
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SYSTEM = (
    "Eres Don Titular. Resumes noticias en exactamente 5 bullets. "
    "Cita el medio entre paréntesis. No inventes hechos que no estén en las notas. "
    "Si un tema no aparece, di 'no aparece'."
)
N_BULLETS = 5
UMBRAL_SOPORTE = 0.5
LARGO_RAIZ = 5

STOPWORDS = frozenset(
    """
    a al algo ante con como contra cual cuando de del desde donde durante el
    ella ellas ellos en entre era es esa ese eso esta este esto estos fue
    ha han hay la las le les lo los mas mientras muy ni no nos o otra otro
    para pero por que se sea segun ser si sin sobre son su sus tambien tras
    un una unas uno unos y ya aparece
    """.split()
)

_RE_NOTA = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")
_RE_BULLET = re.compile(r"^\s*(\d+)[).]\s*(.+?)(?:\s*\(([^)]+)\))?\s*:\s*(.+?)\s*$")
_RE_PALABRA = re.compile(r"[a-zñ]+")
_RE_NUMERO = re.compile(r"\d+")


@dataclass(frozen=True)
class Nota:
    medio: str
    titulo: str
    cuerpo: str = ""

    @property
    def texto(self) -> str:
        return f"{self.titulo} {self.cuerpo}".strip()


@dataclass(frozen=True)
class Bullet:
    numero: int
    tema: str
    medio: str | None
    texto: str

    @property
    def no_aparece(self) -> bool:
        """Solo si el bullet entero es 'no aparece'. 'gratuita (no aparece)' sigue siendo un hecho."""
        return normalizar(self.texto).strip(" .") == "no aparece"


@dataclass
class Juicio:
    bullets: list[Bullet]
    formato_ok: bool
    citas_ok: bool
    soportados: int
    con_hechos: int
    problemas: list[str] = field(default_factory=list)
    citados: int = 0

    @property
    def soporte(self) -> float | None:
        """Fracción de bullets con hechos que tienen soporte. None si no hubo ninguno que medir."""
        return self.soportados / self.con_hechos if self.con_hechos else None

    @property
    def citas(self) -> float | None:
        """Fracción de bullets con hechos que citan un medio de las notas. None si no hubo ninguno."""
        return self.citados / self.con_hechos if self.con_hechos else None

    @property
    def aprobado(self) -> bool:
        return not self.problemas

    def resumen(self) -> str:
        soporte = "—" if self.soporte is None else f"{self.soporte:.0%}"
        marca = {True: "✔", False: "✘"}
        return (
            f"formato {marca[self.formato_ok]} · citas {marca[self.citas_ok]} · "
            f"soporte {soporte} · {len(self.problemas)} problema(s)"
        )


def normalizar(texto: str) -> str:
    """minúsculas, sin acentos (la ñ se queda: 'año' ≠ 'ano')."""
    texto = texto.lower().replace("ñ", "\0")
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return texto.replace("\0", "ñ")


def raices(texto: str) -> set[str]:
    palabras = _RE_PALABRA.findall(normalizar(texto))
    return {p[:LARGO_RAIZ] for p in palabras if len(p) > 2 and p not in STOPWORDS}


def numeros(texto: str) -> set[str]:
    return set(_RE_NUMERO.findall(texto))


def parse_notas(user: str) -> list[Nota]:
    """`N. Medio — Título` + renglones con sangría como cuerpo.

    Sin `—`, el medio queda vacío (así sale `rss_reportero.py` crudo).
    Renglones `fuente: http…` no son contenido y se ignoran.
    """
    notas: list[Nota] = []
    medio, titulo, cuerpo = "", "", []

    def cerrar() -> None:
        if titulo:
            notas.append(Nota(medio, titulo, " ".join(cuerpo)))

    for linea in user.splitlines():
        m = _RE_NOTA.match(linea)
        if m:
            cerrar()
            encabezado = m.group(2)
            medio, _, titulo = encabezado.partition(" — ")
            if not titulo:
                medio, titulo = "", encabezado
            cuerpo = []
        elif titulo and linea.strip() and not linea.strip().startswith("fuente:"):
            cuerpo.append(linea.strip())
    cerrar()
    return notas


def parse_briefing(texto: str) -> list[Bullet]:
    bullets: list[Bullet] = []
    for linea in texto.splitlines():
        m = _RE_BULLET.match(linea)
        if m:
            numero, tema, medio, cuerpo = m.groups()
            bullets.append(Bullet(int(numero), tema.strip(), medio.strip() if medio else None, cuerpo))
    return bullets


def parse_sueltos(texto: str) -> list[Bullet]:
    """Respaldo cuando no hay bullets con formato: cada renglón es un hecho.

    `Medio - texto`, `Medio — texto` o `**Medio**: texto` (≤ 4 palabras antes)
    cuenta como cita. Encabezados sin contenido (`**Resumen:**`) se saltan.
    Así el juez sigue midiendo soporte aunque el modelo ignore el formato.
    """
    bullets: list[Bullet] = []
    for linea in texto.splitlines():
        limpio = re.sub(r"^\s*(?:\d+[).]|[-*•])\s*", "", linea.replace("**", "")).strip()
        if not limpio or limpio.endswith(":"):
            continue
        medio, resto = None, limpio
        partes = re.split(r"\s+[-—]\s+|:\s+", limpio, maxsplit=1)
        if len(partes) == 2 and len(partes[0].split()) <= 4:
            medio, resto = partes
        bullets.append(Bullet(0, "", medio, resto))
    return bullets


def soporte_bullet(bullet: Bullet, contexto: str) -> tuple[float, set[str]]:
    """(fracción de raíces del bullet presentes en el contexto, números inventados)."""
    r = raices(bullet.texto)
    frac = len(r & raices(contexto)) / len(r) if r else 1.0
    return frac, numeros(bullet.texto) - numeros(contexto)


def juzgar(notas: str, briefing: str) -> Juicio:
    """Califica un briefing contra el texto de las notas (el `user`)."""
    lista = parse_notas(notas)
    medios = {normalizar(n.medio) for n in lista if n.medio}
    bullets = parse_briefing(briefing)
    problemas: list[str] = []

    formato_ok = [b.numero for b in bullets] == list(range(1, N_BULLETS + 1))
    if not bullets:
        bullets = parse_sueltos(briefing)
        problemas.append(f"formato: ningún bullet `N) Tema (Medio): …`; se juzgan {len(bullets)} renglones sueltos")
    elif not formato_ok:
        problemas.append(f"formato: se esperaban bullets 1..{N_BULLETS}, llegaron {[b.numero for b in bullets]}")

    vistos: set[str] = set()
    for b in bullets:
        clave = normalizar(b.texto).strip(" .")
        if clave in vistos and not b.no_aparece:
            problemas.append(f"bullet {b.numero}: repite un bullet anterior")
        vistos.add(clave)

    citas_ok, soportados, con_hechos, citados = True, 0, 0, 0
    for b in bullets:
        if b.no_aparece:
            continue
        con_hechos += 1
        quien = f"bullet {b.numero}" if b.numero else f"renglón «{b.texto[:40]}…»"
        if b.medio is None:
            citas_ok = False
            problemas.append(f"{quien}: no cita medio")
        elif medios and normalizar(b.medio) not in medios:
            citas_ok = False
            problemas.append(f"{quien}: cita '{b.medio}', que no está en las notas")
        else:
            citados += 1
        frac, inventados = soporte_bullet(b, notas)
        if inventados:
            problemas.append(f"{quien}: números que no están en las notas {sorted(inventados)}")
        elif frac < UMBRAL_SOPORTE:
            problemas.append(f"{quien}: soporte {frac:.0%} < {UMBRAL_SOPORTE:.0%} — ¿de dónde salió?")
        else:
            soportados += 1

    return Juicio(bullets, formato_ok, citas_ok, soportados, con_hechos, problemas, citados)


def mensajes(notas: str, briefing: str | None = None, system: str = SYSTEM) -> list[dict[str, str]]:
    """Fila de chat. Sin `briefing` es un prompt de inferencia."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": notas}]
    if briefing is not None:
        msgs.append({"role": "assistant", "content": briefing})
    return msgs


def validar_ejemplo(fila: dict) -> list[str]:
    """Regla de oro para el dataset: lista vacía = la fila se queda."""
    por_rol = {m["role"]: m["content"] for m in fila.get("messages", [])}
    faltan = [r for r in ("system", "user", "assistant") if r not in por_rol]
    if faltan:
        return [f"faltan roles {faltan}"]
    return juzgar(por_rol["user"], por_rol["assistant"]).problemas


def notas_desde_items(items: list[dict[str, str]], medio: str, fecha: str, limite: int = 5) -> str:
    """Items de `rss_reportero.parse_items` → bloque `user` con el medio citado."""
    lineas = [f"Notas del {fecha}:"]
    for i, it in enumerate(items[:limite], 1):
        lineas.append(f"{i}. {medio} — {it['title']}")
        if it.get("desc"):
            lineas.append(f"   {it['desc'][:220]}")
    return "\n".join(lineas)


def validar_jsonl(path: Path) -> tuple[int, list[tuple[int, list[str]]]]:
    tirados: list[tuple[int, list[str]]] = []
    total = 0
    for i, linea in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not linea.strip():
            continue
        total += 1
        problemas = validar_ejemplo(json.loads(linea))
        if problemas:
            tirados.append((i, problemas))
    return total, tirados


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "ejemplo-briefing.jsonl")
    total, tirados = validar_jsonl(path)
    for linea, problemas in tirados:
        print(f"renglón {linea}: TIRAR")
        for p in problemas:
            print(f"   - {p}")
    print(f"{total - len(tirados)}/{total} filas pasan la regla de oro ({len(tirados)} tiradas).")
    return 1 if tirados else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
