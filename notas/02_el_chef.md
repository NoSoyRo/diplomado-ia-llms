# 02 · El chef ya existe

Notebooks: [`04_bpe_a_mano`](../parte-2-chef/04_bpe_a_mano.ipynb) · [`05_chat_template_y_villano`](../parte-2-chef/05_chat_template_y_villano.ipynb)

## Idea en una frase

No preentrenas tu propio GPT: descargas un modelo open-weight y aprendes a hablarle con **su** cuchillo (tokenizer) y **su** plato (chat template); sin notas, inventa el día con la misma seguridad con la que dice la verdad.

---

## 1. Tres capas: arquitectura, pretrain, uso

| Capa | Qué es | ¿La haces tú? |
|---|---|---|
| Arquitectura | Transformer decoder (lo del notebook 03, más grande) | No, ya existe |
| Pretrain | Next-token sobre billones de tokens: aquí sale el idioma | No: cuesta millones de dólares |
| Uso / ajuste | Prompt, notas, LoRA | **Sí**: esto es el módulo |

**Open-weight vs API.** Con open-weight (Qwen, Llama, SmolLM) tienes los pesos: puedes correrlo offline, ver los logits y ponerle LoRA. Con una API solo ves texto de salida y pagas por token.

## 2. Tokenizer: del carácter al BPE

**Dummy.** En vez de letras, el modelo ve pedazos frecuentes ("▁la", "ción"). Los pedazos salen de juntar, una y otra vez, la pareja más frecuente. Palabras comunes = un token; palabras raras = varios.

**Matemática.** En cada paso $k$:

$$
(a,b)^\star_k = \arg\max_{(a,b)} \sum_w c(w)\,\#\{i: s_i=a,\ s_{i+1}=b\},
$$

y se reemplaza $a\,b \to ab$ en todas las palabras. $|V| = |V_0| + K$. Con $\bar\ell$ caracteres por token, una ventana de $T$ tokens cubre $\approx T\bar\ell$ caracteres.

**Números del notebook 04** (texto del Quijote no visto): 0 merges → 1.00 chars/token; 1 000 merges → 2.83; Qwen (151 665 tokens) → 3.20. Rendimientos decrecientes: los primeros merges sellan lo que aparece en todos lados.

**Byte-level.** Si la base son los 256 bytes, cualquier texto UTF-8 tiene representación (un emoji = 4 bytes). Sin eso, un carácter no visto no tiene id.

**El tokenizer es parte de $\theta$.** El embedding $E\in\mathbb{R}^{|V|\times d}$ tiene un renglón por token. Cambiar el tokenizer cambia qué significa cada renglón: el modelo queda roto.

## 3. Base vs Instruct

**Dummy.** Mismo cerebro. El base aprendió a continuar texto de internet. El instruct tuvo una segunda escuela con diálogos: sigue instrucciones, a veces se niega.

**Matemática.** Los dos son $P_\theta(x_t\mid x_{<t})$; cambia la distribución del último entrenamiento. El instruct vio textos de la forma $\varphi(\text{system},\text{user})\,\Vert\,\text{assistant}$.

**Lo que viste en el notebook 05.** El base, con "Pregunta: ¿qué pasó hoy? Respuesta:", afirma en greedy que "hoy" pasaron cosas. Con el chat template, suelta una "noticia" de hace años y se cicla: las marcas no le dicen nada.

## 4. Chat template

**Dummy.** El modelo no ve tu lista de mensajes; ve un solo string con marcas especiales. El instruct solo "sabe ser asistente" dentro de ese guion exacto.

```
<|im_start|>system
Eres Don Titular…<|im_end|>
<|im_start|>user
Notas del 2026-08-14: …<|im_end|>
<|im_start|>assistant
```

**Detalles que muerden.**

- `add_generation_prompt=True` deja abierto el turno del assistant.
- Qwen mete un system por default ("You are Qwen…") si no mandas uno.
- Entrenar el LoRA con un formato y servir con otro = hablar con otro modelo.

## 5. El villano: "¿qué pasó hoy?" sin notas

**Dummy.** El modelo no tiene periódico ni reloj. Si le pides las noticias de hoy, la respuesta más probable puede ser negarse; si muestreas, sale algo que *suena* a noticia.

**Matemática.** Sin evidencia, $P_\theta(y\mid q)$ no depende de la fecha real. El greedy aproxima el modo $\arg\max_y P_\theta(y\mid q)$; muestrear con $T$ saca de toda la masa restante. En el notebook 05, con $T=0.5$, el primer token de la negativa ("Lo") tiene 38% y los arranques de una lista ("Aquí", "¡") ya suman 41%: el greedy se niega por un margen chico.

Con notas $d$ se genera de $P_\theta(y\mid q,d)$ y la masa se concentra en continuaciones que copian de $d$. **Eso es RAG casero**: el retriever puede ser un script de RSS.

**Lo que viste.** Con notas, Qwen 0.5B toma los hechos de las notas, pero **no sigue el oficio**: no da 5 bullets, cita a su manera y a veces añade frases ("durante el año pasado"). Eso no se arregla con más notas: se arregla con ejemplos (Parte III).

---

## Autoevaluación

1. ¿Por qué no puedes usar tu BPE del Quijote como tokenizer de Qwen, aunque los dos sean BPE?
2. Con la misma ventana de 4 096 tokens, ¿qué modelo "lee" más texto: uno con 1 000 merges o uno con 150 000? ¿Qué pagas a cambio?
3. El instruct se niega en greedy a darte "las noticias de hoy". ¿Eso demuestra que sabe que no sabe?
4. Mandas mensajes sin system a Qwen. ¿Qué ve el modelo en realidad?
5. ¿Por qué el gym parte del Instruct y no del base?

<details>
<summary>Respuestas</summary>

1. Porque los ids no coinciden: el renglón $i$ del embedding de Qwen corresponde a **su** token $i$. Con otro tokenizer, el id 1 734 apunta a un vector entrenado para otro pedazo.
2. El de 150 000 (más chars/token). Pagas una matriz de salida $d\times|V|$ más grande y un softmax sobre más opciones; tokens raros con embeddings poco entrenados.
3. No. El greedy elige el token más probable; la negativa gana por un margen chico y el resto de la masa son "noticias" inventadas. Basta muestrear para verlas.
4. Un string con el system por default de Qwen ("You are Qwen, created by Alibaba Cloud…") antes de tu user.
5. El Instruct ya sabe el formato de chat y seguir instrucciones; el LoRA solo tiene que enseñar el oficio. Desde el base tendrías que enseñar también a conversar, con muchos más datos.

</details>
