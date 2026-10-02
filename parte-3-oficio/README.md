# Parte III · El oficio: noticiero de las 7 am

Producto: un briefing de 5 bullets que **no inventa**. El día está en el RSS. El estilo está en el LoRA. El juez dice si cumplió.

Notas de estudio: [`notas/03_el_oficio.md`](../notas/03_el_oficio.md).

```
[RSS]  →  rss_reportero.py  →  notas  →  Qwen + LoRA  →  briefing  →  don_titular.py (juez)
```

| Orden | Archivo | Qué es | Corre en |
|---|---|---|---|
| 1 | `06_reportero_y_dataset.ipynb` | Del feed a filas de chat; la regla de oro tira ejemplos | CPU |
| 2 | `07_gym_lora.ipynb` | Entrenar los lentes (llama a `sft_lora_noticias.py`), curva y con/sin lentes | Colab T4 / MPS |
| 3 | `08_juicio_antes_despues.ipynb` | Tabla zero-shot vs few-shot vs LoRA en 10 prompts de hold-out | CPU (+GPU opcional) |

## `rss_reportero.py` — cero IA

**Dummy.** Baja XML y te deja un string con las notas de *hoy*. Si esto falla, el noticiero falla — y no es culpa de Qwen.

**Math.** RAG casero: un retriever $R$ produce $d=R(q)$ y se genera $\hat y\sim P_\theta(\cdot\mid \varphi(\mathrm{system},q,d))$. Es $P(y\mid q,d)$ en vez de $P(y\mid q)$. No hace falta un vector database el día 1.

```bash
python rss_reportero.py
python rss_reportero.py --feed https://feeds.bbci.co.uk/mundo/rss.xml --n 8 --query economia
```

`--n` cuenta las notas que **sobreviven** al filtro, no las primeras n del XML.

## `don_titular.py` — el juez (regla de oro)

**Dummy.** Un briefing pasa si tiene 5 bullets numerados, cada hecho cita un medio que está en las notas, y lo que afirma se puede señalar en las notas. El mismo juez filtra el dataset (antes de entrenar) y califica al modelo (después).

**Math.** Con $S(\cdot)$ = raíces de palabras de contenido y $N(\cdot)$ = números: $\mathrm{soporte}(b,d)=|S(b)\cap S(d)|/|S(b)|$. Un bullet pasa si $\mathrm{soporte}\ge 0.5$ y $N(b)\subseteq N(d)$. Es una cota barata: no detecta "sube" vs "mantiene". El juez humano sigue existiendo.

```bash
python don_titular.py ejemplo-briefing.jsonl     # 44/44 filas pasan
```

## `ejemplo-briefing.jsonl` — dataset Don Titular

Cada renglón es `{"messages": [system, user, assistant]}`. El system pone las reglas, el user trae las notas, el assistant es el briefing canónico. Son **44 filas** (2026-07-01 … 08-19), todas pasan el juez: alcanzan para ver el efecto del LoRA, no para producción. En la sesión 6 escribes 20 de tu oficio, revisadas por ti.

`holdout.jsonl` trae 10 días **posteriores** (08-20 … 08-29) que nunca entran al entrenamiento. `salidas_referencia.json` guarda lo que respondió Qwen en esos 10 días en las tres condiciones (lo regenera `scripts/generar_referencia.py --adapter parte-3-oficio/don-titular-lora`).

## `sft_lora_noticias.py` — SFT + LoRA

**Dummy.** No le haces cirugía al cerebro. Le pones lentes. Los lentes aprenden el *oficio* (5 bullets, cita el medio, di "no aparece"). Las noticias del 14 de agosto **no** viven en el adapter.

**Math.** $W'=W+\frac{\alpha}{r}BA$ con $r\ll\min(d,k)$: entrenas $r(d+k)$ números, no $dk$. SFT es la cross-entropy del lab 0 con máscara solo en el assistant: $\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{\sum m_t}\sum m_t\log P_\theta(x_t\mid x_{<t})$. El script convierte cada fila a `prompt` (system + user) / `completion` (assistant) para que TRL ponga $m_t=0$ en el prompt; con `messages` crudos pagaría también por las notas.

Al terminar escribe `entrenamiento.json` junto al adapter (modelo, filas, épocas, lr, semilla, batch efectivo): es parte del entregable.

```bash
pip install -r ../requirements-oficio.txt
python sft_lora_noticias.py --data ejemplo-briefing.jsonl --epochs 8 --accum 4   # lo que corre el 07
python sft_lora_noticias.py --4bit --data mi-dataset.jsonl     # QLoRA en Colab T4
```
