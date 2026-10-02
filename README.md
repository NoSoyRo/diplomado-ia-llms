# Diplomado IA · Módulo 5 · LLMs

Material para clonar (FI UNAM, 2026-2). Las notas del ebook apuntan aquí.

Ebook (biblioteca): el libro vive en otro repo (`app-web-topics`). Este repo es **solo** notebooks, scripts y PDFs.

```bash
git clone https://github.com/NoSoyRo/diplomado-ia-llms.git
cd diplomado-ia-llms
pip install torch   # labs 1–3 (y 4–6 si los corres)
# oficio (lab LoRA, GPU / Colab T4):
pip install "transformers>=4.45" peft trl datasets accelerate
```

Cada archivo de abajo tiene la misma receta: **intuición dummy**, **una fórmula**, y el detalle en las celdas markdown del notebook (o en el docstring del `.py`).

Notebooks: ábrelos en local o en [Colab desde GitHub](https://colab.research.google.com/github/NoSoyRo/diplomado-ia-llms/blob/main/Ejercicio1LLMs.ipynb). En Colab, descomenta el `%pip` / `wget` de la primera celda de código: el `.txt` no viaja solo.

---

## Parte I · El motor (eje)

El examen del módulo: next-token a mano. Si no puedes señalar la etiqueta, Qwen sigue siendo magia.

### `Ejercicio1LLMs.ipynb` — Lab 0 · GRU / next-token

**Dummy.** Lees en voz alta y, en cada letra, apuestas qué sigue. Ganas si le diste alta probabilidad a la letra que *sí* venía. Nadie etiquetó nada a mano: la respuesta es el texto.

**Math.** Un LM es la regla de la cadena
$P(x_{1:T})=\prod_t P_\theta(x_t\mid x_{<t})$.
Entrenar es minimizar la entropía cruzada
$\mathcal{L}=-\frac{1}{BT}\sum\log P_\theta(y_{b,t}\mid x_{b,\le t})$.
La GRU comprime todo el pasado en un vaso $h_t\in\mathbb{R}^{d}$ de capacidad fija.

Corpus: `quijote.txt` (no `texto.txt`). En CPU el notebook *muestrea* ventanas; no recorre las ~2M.

### `Ejercicio2LLMs.ipynb` — El vaso no basta

**Dummy.** “Más letras atrás ⇒ predice mejor” suena lógico. En una GRU a menudo es falso: le das más tarea al mismo post-it, no un post-it más grande.

**Math.** $I(x_{\le t-L};x_t\mid h_{t-1})\le I(h_{t-1};x_t)$ y $h$ tiene dimensión fija. Alargar $L$ no agranda $d$. Si $\partial\mathcal{L}/\partial h_{t-L}$ se desvanece, el vaso ni usa lo que cabría. Por eso el eje cambia a atención.

### `Ejercicio3LLMs.ipynb` — Mini-Transformer causal

**Dummy.** Cada letra voltea a ver el pasado con pesos (“de lo que ya leí, ¿qué me importa *ahora*?”). Sin máscara causal haces trampa: miras la respuesta. Sin posición, “perro muerde hombre” = “hombre muerde perro”.

**Math.**
$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$,
más $M_{ij}=-\infty$ si $j>i$. Residual $x+f(x)$ para que el gradiente no se muera.

### `quijote.txt`

Corpus de ejemplo (~2M caracteres). Token = un carácter. Transparencia > realismo; BPE llega con Qwen.

---

## Parte III · El oficio (eje)

Producto: briefing de las 7 am que no inventa. El día está en el RSS. El estilo está en el LoRA.

### `rss_reportero.py` — cero IA

**Dummy.** Un script baja XML y te deja un string con las notas de *hoy*. Si esto falla, el noticiero falla — y no es culpa de Qwen. No le preguntes al modelo “qué pasó hoy” sin papel.

**Math.** RAG casero: un retriever $R$ (parsear el RSS) produce $d=R(q)$. Se genera
$\hat y\sim P_\theta(\cdot\mid \varphi(\mathrm{system},q,d))$.
Es $P(y\mid q,d)$ en vez de $P(y\mid q)$. No hace falta un vector database el día 1.

```bash
python rss_reportero.py
python rss_reportero.py --feed https://feeds.bbci.co.uk/mundo/rss.xml --n 8 --query economia
```

### `ejemplo-briefing.jsonl` — dataset Don Titular

**Dummy.** Cada renglón es una obra de un acto: el system pone las reglas (5 bullets, no inventes), el user trae las notas, el assistant escribe el briefing canónico. Si el assistant menciona un terremoto que no está en las notas, **se tira el ejemplo**. El modelo copia lo que ve.

**Math.** Un ejemplo es $(\mathrm{system},q,d,y^\star)$. Regla de oro: cada hecho atómico de $y^\star$ tiene soporte en $d$. Tres filas aquí son la plantilla; el gym pide 200+ revisados por ti.

### `sft_lora_noticias.py` — SFT + LoRA

**Dummy.** No le haces cirugía al cerebro (Qwen). Le pones lentes. Los lentes aprenden el *oficio* (5 bullets, cita el medio, di “no aparece”). Las noticias del 14 de agosto **no** viven en el adapter: siguen viniendo del reportero.

**Math.** LoRA: $W'=W+\frac{\alpha}{r}BA$ con $r\ll\min(d,k)$. Entrenas $r(d+k)$ números, no $dk$. SFT: misma cross-entropy del lab 0, pero la máscara $m_t$ solo enciende tokens del assistant
$\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{\sum m_t}\sum m_t\log P_\theta(x_t\mid x_{<t})$.

```bash
python sft_lora_noticias.py --data ejemplo-briefing.jsonl
# GPU 4-bit (Colab T4):
python sft_lora_noticias.py --4bit --data tus-200.jsonl
```

---

## Extra (temario viejo)

`Ejercicio4–6` no son el eje. Quedan si quieres repetir ingeniería con el juguete:

| Archivo | Dummy | Math |
|---|---|---|
| `Ejercicio4LLMs.ipynb` | El LR no se adivina; se mide. El grid es el experimento más tonto y más honesto. | $\lambda^\star\in\arg\min_{\lambda}\mathcal{L}_{\mathrm{val}}(\theta^\star(\lambda))$ |
| `Ejercicio5LLMs.ipynb` | Más cerebro o más libro bajan la pérdida. No preentrenas GPT-4: compras el pan. | $\mathcal{L}(N,D)\approx E_\infty+(N_c/N)^\alpha+(D_c/D)^\beta$ (Kaplan, ilustrativa) |
| `Ejercicio6LLMs.ipynb` | El T4 no aguanta el batch del paper. Acumulas, mezclas precisión, recomputeas activaciones. | $B_{\mathrm{eff}}=B_{\mathrm{micro}}\times G$ |

---

## PDFs de apoyo (histórico)

`05_01_IntroduccionLLMs.pdf` · `05_02_Entrenamiento_LLMs.pdf` · `05_02_RetoLLM.pdf`

Las notas vivas son el ebook, no estos PDF.
