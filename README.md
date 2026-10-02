# Diplomado IA · Módulo 5 · LLMs

**De cero al noticiero.** Primero construyes el motor a mano (next-token con una GRU y un mini-Transformer). Luego abres un modelo de verdad (Qwen) y ves por qué inventa el día. Al final le pones lentes (LoRA) y un reportero (RSS) para que escriba el briefing de las 7 am **sin inventar**.

FI UNAM · 2026-2. Ebook: el libro vive en otro repo (`app-web-topics`). Este repo es el material para clonar: notebooks, scripts, notas y datos.

```bash
git clone https://github.com/NoSoyRo/diplomado-ia-llms.git
cd diplomado-ia-llms
pip install -r requirements.txt          # Parte I y II (CPU basta)
pip install -r requirements-oficio.txt   # Parte III: LoRA (GPU / Colab T4)
```

Cada notebook sigue la misma receta: **intuición dummy** → **una fórmula** → código que puedes romper → **entregable**. Las [notas](notas/) son el resumen de estudio de cada parte, en el mismo orden.

---

## Ruta

| # | Archivo | Qué aprendes | Corre en |
|---|---|---|---|
| | **[Parte I · El motor](parte-1-motor/)** | | |
| 01 | [`01_next_token_gru.ipynb`](parte-1-motor/01_next_token_gru.ipynb) | La etiqueta es la letra que sigue. Cross-entropy, temperatura. | CPU |
| 02 | [`02_el_vaso_no_basta.ipynb`](parte-1-motor/02_el_vaso_no_basta.ipynb) | Más contexto no ayuda si el estado es de tamaño fijo. | CPU |
| 03 | [`03_mini_transformer.ipynb`](parte-1-motor/03_mini_transformer.ipynb) | Atención, máscara causal, posición. | CPU |
| | **[Parte II · El chef ya existe](parte-2-chef/)** | | |
| 04 | [`04_bpe_a_mano.ipynb`](parte-2-chef/04_bpe_a_mano.ipynb) | El tokenizer se entrena: BPE desde cero sobre el Quijote. | CPU, sin torch |
| 05 | [`05_chat_template_y_villano.ipynb`](parte-2-chef/05_chat_template_y_villano.ipynb) | Qwen real: base vs instruct, chat template, el villano que inventa el día. | CPU (~1 min, baja 2 GB) / Colab |
| | **[Parte III · El oficio](parte-3-oficio/)** | | |
| 06 | [`06_reportero_y_dataset.ipynb`](parte-3-oficio/06_reportero_y_dataset.ipynb) | RSS → notas → filas de chat. La regla de oro tira ejemplos. | CPU, sin torch |
| — | [`sft_lora_noticias.py`](parte-3-oficio/sft_lora_noticias.py) | SFT + LoRA: los lentes de Don Titular. | GPU (T4) |
| 07 | [`07_juicio_antes_despues.ipynb`](parte-3-oficio/07_juicio_antes_despues.ipynb) | Tabla antes vs después con un juez reproducible. | CPU, sin torch |
| | **[Reto](reto/)** | Tu oficio, no el nuestro. Checklist de entrega. | |

Extras de la Parte I (temario viejo, opcionales): [`parte-1-motor/extra/`](parte-1-motor/extra/) — grid de hiperparámetros, scaling, memoria/VRAM.

## Estructura

```
diplomado-ia-llms/
├── notas/              apuntes por parte + fórmulas + glosario
├── parte-1-motor/      01–03 (+ extra/ E1–E3)
├── parte-2-chef/       04–05
├── parte-3-oficio/     06–07, rss_reportero.py, don_titular.py, sft_lora_noticias.py, ejemplo-briefing.jsonl
├── reto/               enunciado y checklist del reto final
├── datos/quijote.txt   corpus de la Parte I y II (~2M caracteres)
├── pdfs/               PDFs históricos del temario (las notas vivas son el ebook)
└── tests/              pytest del reportero y del juez (corre en CI)
```

## Colab

Abre cualquier notebook con `https://colab.research.google.com/github/NoSoyRo/diplomado-ia-llms/blob/main/<ruta>`, por ejemplo [01 en Colab](https://colab.research.google.com/github/NoSoyRo/diplomado-ia-llms/blob/main/parte-1-motor/01_next_token_gru.ipynb).

Colab solo baja el notebook. La primera celda de código trae comentados el `%pip` y el `wget` de lo que ese notebook necesita (el corpus o los `.py` de la Parte III): descoméntalos una vez.

## Las tres piezas (no se borra del pizarrón)

```
[RSS / API]  →  notas de HOY  →  LLM (+ LoRA)  →  briefing
 reportero        el papel         conductor
```

¿Dónde está **el día**? En el feed. ¿Dónde está **el estilo**? En los lentes. ¿Dónde está **el idioma**? En el pretrain que no hiciste tú. Si confundes las tres, el reto se cae.

## Para el profe / CI

```bash
pip install -r requirements-dev.txt
pytest -q                                   # juez + reportero
python scripts/run_notebooks.py             # ejecuta 04, 06, 07 (sin red ni GPU)
python parte-3-oficio/don_titular.py parte-3-oficio/ejemplo-briefing.jsonl
```
