# Parte II · El chef ya existe

En la Parte I cocinaste con un juguete que habla "quijotesco" letra por letra. Nadie en 2026 preentrena su propio GPT: descargas un chef que ya existe (Qwen, SmolLM, Llama) y aprendes **cómo se le habla**. Esta parte es el puente: el cuchillo (tokenizer), el plato (chat template) y el villano (inventa el día si no le das papel).

Notas de estudio: [`notas/02_el_chef.md`](../notas/02_el_chef.md).

## `04_bpe_a_mano.ipynb` — El cuchillo del modelo

**Dummy.** En la Parte I cada letra era un token. Qwen no ve letras: ve pedazos ("▁la", "ción", "Mancha"). Esos pedazos no los eligió una persona; salen de juntar, una y otra vez, el par de pedazos más frecuente. Lo programas tú, sobre el Quijote, en Python puro.

**Math.** BPE es codicioso: en cada paso $k$, $(a,b)^\star=\arg\max_{(a,b)} \mathrm{cuenta}(a,b)$ y se reemplaza por un símbolo nuevo $ab$. Con $K$ merges el vocabulario crece a $|V_0|+K$ y la longitud baja. Para la misma ventana de $T$ tokens el modelo ve más texto: $\text{caracteres vistos}\approx T\cdot\overline{\text{chars/token}}$.

**Entregable.** Tabla `merges → chars/token` y un ejemplo donde el tokenizer del Quijote corta mal una palabra moderna ("Banxico").

## `05_chat_template_y_villano.ipynb` — Qwen real

**Dummy.** Un *Instruct* no es un cerebro distinto: es el mismo next-token, entrenado para continuar un guion de chat con marcas especiales. Si le preguntas "¿qué pasó hoy?" sin notas, continúa el guion con lo más probable — que no es lo de hoy. Con las notas pegadas, deja de inventar tanto. Eso ya es RAG casero.

**Math.** Sin evidencia, el modo $\arg\max_y P_\theta(y\mid q)$ es lo típico del pretrain, no el día. Con notas, $P_\theta(y\mid q,d)$ concentra masa en lo que $d$ dice. El chat template $\varphi$ es parte del modelo: otro formato = otra distribución.

**Entregable.** Dos salidas guardadas (sin notas / con notas) calificadas por el juez `don_titular.py`. Esa captura es el "antes" del reto.
