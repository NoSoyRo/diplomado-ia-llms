# 03 · El oficio: noticiero de las 7 am

Notebooks y scripts: [`06_reportero_y_dataset`](../parte-3-oficio/06_reportero_y_dataset.ipynb) · [`07_gym_lora`](../parte-3-oficio/07_gym_lora.ipynb) · [`08_juicio_antes_despues`](../parte-3-oficio/08_juicio_antes_despues.ipynb) · [`sft_lora_noticias.py`](../parte-3-oficio/sft_lora_noticias.py) · [`don_titular.py`](../parte-3-oficio/don_titular.py)

## Idea en una frase

El día lo trae un script (reportero), el oficio lo aprende un parche barato (LoRA) a partir de ejemplos limpios (regla de oro), y un juez reproducible más tus ojos dicen si funcionó.

---

## 1. La arquitectura del producto

```
[RSS / API]  →  rss_reportero.py  →  notas  →  Qwen + LoRA  →  briefing  →  don_titular.py
 reportero        (cero IA)          el papel     conductor                    juez
```

| Pregunta | Respuesta |
|---|---|
| ¿Dónde está el día? | En el feed. Si el feed falla, no hay noticiero (y el sistema debe decirlo). |
| ¿Dónde está el estilo? | En el adapter LoRA. |
| ¿Dónde está el idioma? | En el pretrain de Qwen. |

**Matemática (RAG casero).** Un retriever $R$ produce $d=R(q)$ y se genera $\hat y \sim P_\theta(\cdot\mid\varphi(s,q,d))$. El día 1 no hace falta un vector database: un string basta.

**Detalle del reportero.** `--n` cuenta las notas que **sobreviven** al filtro `--query`. Si recortaras primero y filtraras después, `--query economia` podría salir vacío aunque más abajo hubiera economía.

## 2. El dataset es el trabajo

**Dummy.** El modelo copia lo que ve. Un ejemplo donde el assistant inventa le enseña que inventar se vale. Se tira, no se corrige.

**Matemática.** Una fila es $(s,q,d,y^\star)$. Regla de oro: $\forall h\in\mathcal{H}(y^\star),\ d\models h$. Una fila mala hace que el gradiente suba $P_\theta(h\mid s,q,d)$ para un hecho sin soporte.

**Hold-out por fecha.** Si el mismo día está en train y en prueba, el modelo ya vio esas notas. Separa por fecha, no al azar.

**Cuánto.** El ejemplo trae 44 filas: basta para que el LoRA aprenda el formato, no para que cite siempre (sección 6). En la sesión 6 escribes 20 de **tu** oficio; para que un LoRA cite de verdad harían falta cientos. Incluye filas con "no aparece": si nunca lo practica, nunca lo dice.

## 3. El juez (`don_titular.py`)

Tres indicadores, sin IA:

- **Formato**: bullets `1) … 5)`.
- **Citas**: cada hecho cita un medio que está en las notas (se reporta como fracción de bullets bien citados).
- **Soporte**: con $S(\cdot)$ = raíces de palabras de contenido y $N(\cdot)$ = números,

$$
\mathrm{soporte}(b,d) = \frac{|S(b)\cap S(d)|}{|S(b)|}\ \ge\ 0.5 \quad\text{y}\quad N(b)\subseteq N(d).
$$

**Lo que no ve.** "Banxico *baja* la tasa" usa las mismas palabras que "mantiene la tasa": pasa. Atribuir un hecho al medio equivocado: pasa si el medio está en las notas. Un "Deportes: no aparece" en un día con tres notas de deportes: pasa. El juez filtra lo obvio a escala; tú lees lo sutil.

## 4. LoRA: lentes, no cirugía

**Dummy.** No reentrenas los 494 M de parámetros de Qwen: se te olvida el idioma, no cabe en Colab. Congelas todo y entrenas un parche delgado que se suma a algunas matrices. Quitas el parche y vuelves al Qwen original.

**Matemática.** Para una matriz congelada $W\in\mathbb{R}^{d\times k}$:

$$
W' = W + \frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k).
$$

Entrenas $r(d+k)$ números en vez de $dk$. $B$ se inicializa en cero: al arrancar, $W'=W$.

**Cuenta real** (Qwen2.5-0.5B, `r=16` en `q_proj` y `v_proj`, como en `sft_lora_noticias.py`):

| Matriz | Forma | Params LoRA por capa |
|---|---|---|
| `q_proj` | 896 × 896 | 16 · (896 + 896) = 28 672 |
| `v_proj` | 128 × 896 (atención GQA: 2 cabezas KV × 64) | 16 · (128 + 896) = 16 384 |
| **24 capas** | | **1 081 344** |

Contra 494 032 768 parámetros del modelo: **0.22%**. El adapter pesa unos MB, no 1 GB.

**QLoRA.** $W$ se guarda en 4 bits (~4× menos memoria que fp16) y se descuantiza al vuelo; $A$, $B$ siguen en bf16. Es lo que hace caber modelos de 1–2 B en un T4 de Colab.

## 5. SFT: next-token solo del assistant

**Dummy.** Es el mismo juego del notebook 01, pero el modelo solo paga por equivocarse en el briefing. Las notas son contexto: se leen, no se practican. En TRL eso se logra pasando cada fila como `prompt` / `completion`; con `messages` crudos la pérdida cubre todo.

**Matemática.**

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{\sum_t m_t}\sum_t m_t\log P_\theta(x_t\mid x_{<t}),\qquad m_t=\mathbb{1}[t\in\text{assistant}].
$$

En la fila de ejemplo, $m_t=1$ en ~34% de los caracteres (notebook 06).

**Batch efectivo.** $B_{\text{eff}} = B_{\text{micro}}\times G$ (acumulación de gradiente). El script usa $2\times 8=16$ por defecto; con 44 filas el notebook 07 baja a $2\times 4=8$ y sube a 8 épocas para que haya suficientes pasos (~40).

## 6. El juicio: antes vs después

**Condiciones** (notebook 08): `zero-shot` (system + notas), `few-shot` (+ un ejemplo resuelto en el prompt), `lora` (system + notas, con el adapter del 07).

**Lo que dio la corrida guardada** (Qwen2.5-0.5B-Instruct, greedy, 10 prompts de hold-out; LoRA r=16, 44 filas, 8 épocas):

| | zero-shot | few-shot | lora |
|---|---|---|---|
| formato (5 bullets) | 0/10 | 1/10 | **10/10** |
| bullets bien citados | 15/28 | 8/28 | 15/28 |
| soporte medio | 89% | 100% | 100% |
| problemas del juez | 26 | 29 | **13** |

El modelo **lee** bien las notas en las tres condiciones. El LoRA aprendió la **forma** (lo que ningún prompt logró) y no el hábito de citar: con 44 filas, SFT enseña primero el formato.

**Tu LoRA tiene que ganarle a few-shot.** Si no le gana, un prompt con un ejemplo era más barato.

**Goodhart.** Si optimizas para el juez, el modelo aprende a satisfacerlo. En la corrida guardada, el LoRA saca **0 problemas** en h10 y escribe "Viaducto (ESPN): cierre parcial" (el cierre era de *Local*): medios reales, palabras reales, atribución rota. La columna humana existe por eso.

---

## Autoevaluación

1. ¿Las noticias del 14 de agosto están en el adapter? ¿Dónde están?
2. Una fila tiene un briefing perfecto salvo un bullet "Sismo en Oaxaca" que no está en las notas. ¿La corriges o la tiras? ¿Por qué?
3. ¿Cuántos parámetros entrena LoRA con `r=8` en `q_proj` y `v_proj` de Qwen2.5-0.5B?
4. ¿Por qué se separa el hold-out por fecha?
5. Tu LoRA saca formato 100% y soporte 100% en el juez, pero tú no publicarías ningún briefing. ¿Qué pasó?

<details>
<summary>Respuestas</summary>

1. No. El adapter guarda el oficio (forma, citas, "no aparece"). Las noticias están en las notas que trae el reportero en cada llamada.
2. Se tira y se cuenta. Corregir a mano tiende a dejar rastros; además, contar cuántas tiras es parte del entregable y mide qué tan sucia es tu fuente.
3. La mitad que con `r=16`: $8\cdot(896+896) + 8\cdot(128+896) = 22\,528$ por capa, × 24 = **540 672**.
4. Para que el modelo no haya visto las mismas notas en entrenamiento: si no, el "después" se infla.
5. Goodhart: el modelo aprendió a satisfacer al juez (copiar títulos, rellenar "no aparece") sin resumir. El juez es una cota barata, no la meta.

</details>
