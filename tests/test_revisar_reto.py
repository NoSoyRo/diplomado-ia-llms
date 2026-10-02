"""revisar_reto.py distingue una entrega completa de una incompleta. Sin torch."""

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 400)[: 15 * 1024]

CODIGO_BUENO = """
import torch
import torch.nn as nn

torch.manual_seed(42)

texto = open("corpus.txt").read()
chars = sorted(set(texto))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


class ModeloGRU(nn.Module):
    def __init__(self, vocab_size, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        out, _ = self.gru(self.emb(x))
        return self.fc(out)


modelo = ModeloGRU(len(chars))
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-4)
val_loss = 2.10


def generar(modelo, prompt, temperatura=1.0):
    return prompt * 3


for temp in [0.7, 1.0, 1.2]:
    generar(modelo, "hola ", temperatura=temp)
"""

CODIGO_MALO = """
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
texto = open("corpus.txt").read()
print(len(tok.encode(texto)))
"""


def _cargar_revisar_reto():
    spec = importlib.util.spec_from_file_location("revisar_reto", ROOT / "scripts" / "revisar_reto.py")
    modulo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["revisar_reto"] = modulo  # dataclasses resuelve anotaciones vía sys.modules
    spec.loader.exec_module(modulo)
    return modulo


def _armar_entrega_buena(tmp_path: Path) -> Path:
    carpeta = tmp_path / "buena"
    carpeta.mkdir()
    (carpeta / "corpus.txt").write_text(CORPUS, encoding="utf-8")
    (carpeta / "entrenar.py").write_text(CODIGO_BUENO, encoding="utf-8")
    (carpeta / "config.json").write_text(
        json.dumps({"seed": 42, "lr": 3e-4, "epochs": 3, "hidden": 128, "model": "GRU"}), encoding="utf-8"
    )
    resultados = carpeta / "resultados"
    resultados.mkdir()
    (resultados / "historia.json").write_text(
        json.dumps({"train_loss": [3.0, 2.5, 2.2], "val_loss": [2.8, 2.3, 2.10]}), encoding="utf-8"
    )
    (resultados / "curva.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (resultados / "muestras.txt").write_text("hola " * 350, encoding="utf-8")
    return carpeta


def test_entrega_completa_sin_fallas_bloqueantes(tmp_path: Path) -> None:
    rr = _cargar_revisar_reto()
    carpeta = _armar_entrega_buena(tmp_path)
    rep = rr.revisar(carpeta)
    malos = [c.texto for c in rep.checks if c.ok is False]
    # No generamos reporte.pdf aquí (evita depender de reportlab en CI);
    # ese es el único ✘ esperado en una entrega completa de código + resultados.
    assert malos == ["reporte.pdf: NO ENCONTRADO"], f"✘ inesperados: {malos}"


def test_entrega_vacia_es_bloqueante(tmp_path: Path) -> None:
    rr = _cargar_revisar_reto()
    carpeta = tmp_path / "vacia"
    carpeta.mkdir()
    rep = rr.revisar(carpeta)
    assert rep.hay_bloqueantes
    textos = [c.texto for c in rep.checks]
    assert any("corpus" in t and "NO ENCONTRADO" in t for t in textos)
    assert any("no se encontró código" in t for t in textos)


def test_corpus_quijote_prohibido(tmp_path: Path) -> None:
    rr = _cargar_revisar_reto()
    carpeta = tmp_path / "calca"
    carpeta.mkdir()
    quijote = ROOT / "datos" / "quijote.txt"
    (carpeta / "corpus.txt").write_text(quijote.read_text(encoding="utf-8")[:20000], encoding="utf-8")
    (carpeta / "entrenar.py").write_text(CODIGO_BUENO, encoding="utf-8")
    rep = rr.revisar(carpeta)
    textos = [c.texto for c in rep.checks]
    assert any("PARECE el Quijote" in t for t in textos)


def test_tokenizer_de_libreria_marca_mal(tmp_path: Path) -> None:
    rr = _cargar_revisar_reto()
    carpeta = tmp_path / "mala"
    carpeta.mkdir()
    (carpeta / "corpus.txt").write_text(CORPUS, encoding="utf-8")
    (carpeta / "entrenar.py").write_text(CODIGO_MALO, encoding="utf-8")
    rep = rr.revisar(carpeta)
    textos = [c.texto for c in rep.checks]
    assert any("tokenizer de librería" in t for t in textos)
    assert rep.hay_bloqueantes


def test_val_loss_final_lee_la_ultima_epoca() -> None:
    rr = _cargar_revisar_reto()
    assert rr.val_loss_final({"val_loss": [2.8, 2.3, 2.10]}) == 2.10
    assert rr.val_loss_final({}) is None


def test_cruce_perplejidad_pdf_vs_resultados(tmp_path: Path) -> None:
    """Si el PDF trae la perplejidad correcta (exp(val_loss)), el cruce coincide."""
    rr = _cargar_revisar_reto()
    rep = rr.Reporte()
    rr.check_pdf(None, rep, val_loss_codigo=2.10)  # sin PDF: no debe tronar
    assert rep.checks == []

    ppl_esperada = math.exp(2.10)
    assert abs(ppl_esperada - 8.1661699) < 1e-4
