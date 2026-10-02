"""Revisión automática del reto (mini-LLM por carácter). Lee código + reporte.pdf.

No pone la nota: junta evidencia para que un humano (o el AGENTE_reto_llms.md)
decida. Cada check es una línea con ✔ / ✘ / ⚠ y la cita (archivo, línea, número).

Uso
---
    python scripts/revisar_reto.py <carpeta_de_la_entrega> [--json]

La carpeta esperada (ver reto/README.md):

    mi-mini-llm/
    ├── corpus.txt
    ├── entrenar.py        (o *.ipynb)
    ├── config.json
    ├── resultados/        historia.json|csv, *.png, muestras.txt
    └── reporte.pdf

No ejecuta el entrenamiento: eso puede tardar minutos u horas según el corpus.
Lee lo que el alumno ya generó y lo cruza contra el reporte.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_MIN_KB = 10
CORPUS_MAX_KB = 500
GEN_TOKENS_MIN = 300
GEN_TOKENS_MAX = 600
N_TEMPERATURAS_MIN = 3
SECCIONES_REPORTE = [
    ("datos", r"\bdatos?\b"),
    ("problema / etiqueta", r"next[\s-]?token|etiqueta|cadena"),
    ("modelo", r"\bmodelo\b|arquitectura|transformer|\bgru\b|\brnn\b"),
    ("entrenamiento", r"entrenamiento|hiperpar[aá]metros"),
    ("resultados", r"resultados?|perplejidad|val_?loss"),
    ("generaci[oó]n", r"generaci[oó]n|temperatura"),
    ("limitaciones", r"limitaci[oó]n|mejora"),
]


@dataclass
class Check:
    ok: bool | None  # True ✔, False ✘, None ⚠ (no se pudo verificar)
    texto: str


@dataclass
class Reporte:
    checks: list[Check] = field(default_factory=list)

    def add(self, ok: bool | None, texto: str) -> None:
        self.checks.append(Check(ok, texto))

    def imprimir(self) -> None:
        simbolo = {True: "✔", False: "✘", None: "⚠"}
        for c in self.checks:
            print(f"{simbolo[c.ok]} {c.texto}")

    def a_dict(self) -> dict:
        return {"checks": [{"ok": c.ok, "texto": c.texto} for c in self.checks]}

    @property
    def hay_bloqueantes(self) -> bool:
        return any(c.ok is False for c in self.checks)


def leer_codigo(carpeta: Path) -> str:
    """Concatena todo el .py (y las celdas de código de cualquier .ipynb)."""
    partes: list[str] = []
    for p in sorted(carpeta.rglob("*.py")):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        partes.append(f"# ==== {p.name} ====\n{p.read_text(encoding='utf-8', errors='replace')}")
    for p in sorted(carpeta.rglob("*.ipynb")):
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for celda in nb.get("cells", []):
            if celda.get("cell_type") == "code":
                src = celda.get("source", "")
                partes.append("".join(src) if isinstance(src, list) else src)
    return "\n\n".join(partes)


def check_estructura(carpeta: Path, rep: Reporte) -> dict[str, Path | None]:
    archivos = {
        "corpus": next(iter(carpeta.rglob("corpus.txt")), None) or next(iter(carpeta.rglob("*.txt")), None),
        "config": next(iter(carpeta.rglob("config.json")), None),
        "reporte_pdf": next(iter(carpeta.rglob("reporte.pdf")), None) or next(iter(carpeta.rglob("*.pdf")), None),
        "resultados": next((p for p in carpeta.rglob("*") if p.is_dir() and p.name == "resultados"), None),
    }
    rep.add(archivos["corpus"] is not None, f"corpus: {archivos['corpus'] or 'NO ENCONTRADO'}")
    rep.add(archivos["config"] is not None, f"config.json: {archivos['config'] or 'NO ENCONTRADO'}")
    rep.add(archivos["reporte_pdf"] is not None, f"reporte.pdf: {archivos['reporte_pdf'] or 'NO ENCONTRADO'}")
    rep.add(archivos["resultados"] is not None, f"carpeta resultados/: {archivos['resultados'] or 'NO ENCONTRADA'}")
    return archivos


def check_corpus(corpus: Path | None, rep: Reporte) -> None:
    if corpus is None:
        return
    kb = corpus.stat().st_size / 1024
    ok_tam = CORPUS_MIN_KB <= kb <= CORPUS_MAX_KB
    rep.add(ok_tam, f"corpus.txt: {kb:.1f} KB (rango pedido {CORPUS_MIN_KB}-{CORPUS_MAX_KB} KB)")
    texto = corpus.read_text(encoding="utf-8", errors="replace")
    marcas_quijote = ("don Quijote de la Mancha", "Sancho Panza", "Alonso Quijano", "Dulcinea del Toboso")
    es_quijote = any(m in texto for m in marcas_quijote)
    rep.add(
        not es_quijote,
        "corpus.txt no es el Quijote del curso"
        if not es_quijote
        else "corpus.txt PARECE el Quijote del curso (prohibido)",
    )


def check_config(config: Path | None, rep: Reporte) -> dict:
    if config is None:
        return {}
    try:
        datos = json.loads(config.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        rep.add(False, f"config.json no es JSON válido: {e}")
        return {}
    texto = json.dumps(datos).lower()
    tiene_seed = bool(re.search(r"seed|semilla", texto))
    rep.add(
        tiene_seed, "config.json declara una semilla" if tiene_seed else "config.json: no se ve una semilla declarada"
    )
    claves_esperadas = ("lr", "learning_rate", "epochs", "seq_len", "hidden", "n_embd", "batch")
    presentes = [k for k in claves_esperadas if k in texto]
    rep.add(len(presentes) >= 3, f"config.json trae hiperparámetros: {presentes or 'casi ninguno'}")
    return datos


def _llama_a(nodo: ast.AST, nombres: tuple[str, ...]) -> bool:
    for n in ast.walk(nodo):
        if isinstance(n, ast.Call):
            f = n.func
            nombre = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if nombre in nombres:
                return True
    return False


def check_codigo(codigo: str, rep: Reporte) -> None:
    try:
        arbol = ast.parse(codigo)
    except SyntaxError as e:
        rep.add(False, f"el código no parsea: {e}")
        return

    # Tokenización por carácter: debe existir un vocabulario construido desde el texto,
    # no un tokenizer de librería (AutoTokenizer, tiktoken, SentencePiece) sobre el corpus propio.
    usa_tokenizer_externo = bool(re.search(r"AutoTokenizer|tiktoken|sentencepiece|BertTokenizer", codigo))
    tiene_stoi = bool(re.search(r"\b(stoi|char2idx|vocab|char_to_id)\b\s*=", codigo)) or bool(
        re.search(r"enumerate\(sorted\(set\(", codigo)
    )
    rep.add(
        tiene_stoi and not usa_tokenizer_externo,
        "vocabulario por carácter (stoi/itos o equivalente) construido a mano"
        if tiene_stoi and not usa_tokenizer_externo
        else f"tokenización: {'usa un tokenizer de librería sobre el corpus propio' if usa_tokenizer_externo else 'no se encontró stoi/itos explícito'}",
    )

    # Modelo: GRU/LSTM/RNN (baseline) o atención con máscara causal (avanzado).
    clases = [n.name for n in ast.walk(arbol) if isinstance(n, ast.ClassDef)]
    usa_gru = bool(re.search(r"\bnn\.(GRU|LSTM|RNN)\b", codigo))
    usa_atencion = bool(re.search(r"attention|self_attn|qkv|n_head", codigo, re.IGNORECASE))
    rep.add(
        usa_gru or usa_atencion,
        f"modelo: {'GRU/LSTM/RNN' if usa_gru else 'atención' if usa_atencion else 'no identificado'} (clases: {clases or 'ninguna'})",
    )
    if usa_atencion and not usa_gru:
        causal = bool(re.search(r"\btril\b|masked_fill|causal.?mask|is_causal\s*=\s*True", codigo))
        rep.add(
            causal,
            "máscara causal presente" if causal else "Transformer SIN evidencia de máscara causal (tril / masked_fill)",
        )

    # Entrenamiento.
    cross_entropy = bool(re.search(r"CrossEntropyLoss|cross_entropy", codigo))
    rep.add(
        cross_entropy, "usa entropía cruzada" if cross_entropy else "no se encontró CrossEntropyLoss / cross_entropy"
    )
    optimizador = bool(re.search(r"optim\.Adam(W)?\(", codigo))
    rep.add(optimizador, "usa Adam/AdamW" if optimizador else "no se encontró torch.optim.Adam(W)")
    registra_val = bool(re.search(r"val_loss|eval_loss|validation", codigo, re.IGNORECASE))
    rep.add(
        registra_val,
        "registra una pérdida de validación" if registra_val else "no se ve val_loss / validación en el código",
    )

    # Generación.
    tiene_generate = bool(re.search(r"def\s+generat\w*|def\s+generar\w*", codigo))
    rep.add(
        tiene_generate,
        "tiene una función de generación" if tiene_generate else "no se encontró una función generate/generar",
    )
    temps = {float(x) for x in re.findall(r"temperatur\w*\s*[=:]\s*([\d.]+)", codigo)}
    temps |= {
        float(x) for m in re.findall(r"\[([\d.,\s]+)\]", codigo) for x in re.findall(r"[\d.]+", m) if 0 < float(x) <= 3
    }
    rep.add(
        len(temps) >= N_TEMPERATURAS_MIN or None,
        f"temperaturas detectadas en el código: {sorted(temps) if temps else 'ninguna (puede estar en config.json o en resultados/)'}",
    )

    # Reproducibilidad.
    tiene_seed_call = bool(re.search(r"manual_seed\(|seed_everything\(|np\.random\.seed\(", codigo))
    rep.add(
        tiene_seed_call,
        "fija semilla en el código (manual_seed / seed)" if tiene_seed_call else "no se ve una llamada a fijar semilla",
    )

    # Negativo: el notebook del curso pegado tal cual (reutiliza nombres muy específicos sin adaptar).
    huellas_curso = ("CharWindowDataset", "CharGRULM", "MiniTransformerLM")
    calca = sum(h in codigo for h in huellas_curso)
    if calca >= 2 and "quijote" in codigo.lower():
        rep.add(False, "el código parece el notebook 01/02/03 sin adaptar (clases y Quijote del curso)")


def check_resultados(resultados: Path | None, rep: Reporte) -> dict:
    datos: dict = {}
    if resultados is None:
        return datos
    historia = next(iter(resultados.glob("historia*.json")), None) or next(iter(resultados.glob("historia*.csv")), None)
    rep.add(historia is not None, f"log de train/val: {historia or 'no encontrado (historia.json o .csv)'}")
    figuras = list(resultados.glob("*.png")) + list(resultados.glob("*.jpg"))
    rep.add(len(figuras) >= 1, f"figura de la curva: {[f.name for f in figuras] or 'ninguna'}")
    muestras = next(iter(resultados.glob("muestras*.txt")), None) or next(
        iter(resultados.glob("generacion*.txt")), None
    )
    if muestras:
        texto = muestras.read_text(encoding="utf-8", errors="replace")
        n = len(texto.split())
        rep.add(None, f"muestras de generación: {muestras.name} ({n} palabras, revisar longitud 300-600 tokens a ojo)")
    else:
        rep.add(None, "no se encontró muestras*.txt / generacion*.txt en resultados/ (puede estar solo en el PDF)")

    if historia and historia.suffix == ".json":
        try:
            datos = json.loads(historia.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return datos


def val_loss_final(historia: dict) -> float | None:
    for clave in ("val_loss", "val"):
        serie = historia.get(clave)
        if isinstance(serie, list) and serie:
            return float(serie[-1])
    return None


def extraer_texto_pdf(pdf: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        lector = PdfReader(str(pdf))
        return "\n".join(pagina.extract_text() or "" for pagina in lector.pages)
    except Exception:  # noqa: BLE001 — un PDF roto no debe tronar la revisión
        return ""


def check_pdf(pdf: Path | None, rep: Reporte, val_loss_codigo: float | None) -> None:
    if pdf is None:
        return
    texto = extraer_texto_pdf(pdf)
    if not texto.strip():
        rep.add(False, "reporte.pdf: sin texto extraíble (parece imagen / escaneado)")
        return
    rep.add(True, f"reporte.pdf: {len(texto)} caracteres de texto extraído")

    portada = texto[:800]
    tiene_cuenta = bool(re.search(r"cuenta|matr[ií]cula|\bid\b", portada, re.IGNORECASE))
    tiene_numero = bool(re.search(r"\d{5,}", portada))
    rep.add(
        tiene_cuenta and tiene_numero or None,
        "portada: se ve nombre/cuenta"
        if (tiene_cuenta and tiene_numero)
        else "portada: no se identifica claramente cuenta/ID (revisar a mano)",
    )

    bajo = texto.lower()
    faltan = [nombre for nombre, patron in SECCIONES_REPORTE if not re.search(patron, bajo)]
    rep.add(
        not faltan,
        "las 8 secciones tienen rastro en el texto" if not faltan else f"secciones sin rastro en el PDF: {faltan}",
    )

    if re.search(r"faltó tiempo|falta de tiempo|no hubo tiempo", bajo):
        rep.add(False, "limitaciones: dice 'faltó tiempo' (la rúbrica la pone en 0)")

    ppl_en_pdf = [float(x) for x in re.findall(r"perplej\w*[^\d]{0,20}([\d]+\.?[\d]*)", texto, re.IGNORECASE)]
    if val_loss_codigo is not None and ppl_en_pdf:
        ppl_esperada = math.exp(val_loss_codigo)
        coincide = any(abs(p - ppl_esperada) / max(ppl_esperada, 1e-6) < 0.15 for p in ppl_en_pdf)
        rep.add(
            coincide,
            f"perplejidad del PDF {ppl_en_pdf} vs exp(val_loss) de resultados/ = {ppl_esperada:.2f}"
            + ("" if coincide else " → NO coinciden, revisar de dónde salió el número"),
        )
    elif ppl_en_pdf:
        rep.add(None, f"perplejidad mencionada en el PDF: {ppl_en_pdf} (no hay historia.json para cruzar)")
    else:
        rep.add(False, "no se encontró un número de perplejidad en el texto del PDF")


def revisar(carpeta: Path) -> Reporte:
    rep = Reporte()
    archivos = check_estructura(carpeta, rep)
    check_corpus(archivos["corpus"], rep)
    check_config(archivos["config"], rep)
    codigo = leer_codigo(carpeta)
    if codigo.strip():
        check_codigo(codigo, rep)
    else:
        rep.add(False, "no se encontró código (.py ni .ipynb) en la carpeta")
    historia = check_resultados(archivos["resultados"], rep)
    check_pdf(archivos["reporte_pdf"], rep, val_loss_final(historia))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", type=Path, help="carpeta de la entrega del reto")
    ap.add_argument("--json", action="store_true", help="salida en JSON en vez de texto")
    args = ap.parse_args()

    if not args.carpeta.is_dir():
        print(f"no existe la carpeta: {args.carpeta}", file=sys.stderr)
        return 2

    rep = revisar(args.carpeta)
    if args.json:
        print(json.dumps(rep.a_dict(), ensure_ascii=False, indent=2))
    else:
        rep.imprimir()
        n_ok = sum(c.ok is True for c in rep.checks)
        n_mal = sum(c.ok is False for c in rep.checks)
        n_dudoso = sum(c.ok is None for c in rep.checks)
        print(f"\n{n_ok} ✔ · {n_mal} ✘ · {n_dudoso} ⚠ — esto no es la nota, es evidencia para AGENTE_reto_llms.md")
    return 1 if rep.hay_bloqueantes else 0


if __name__ == "__main__":
    sys.exit(main())
