# 01 · El motor

Notebooks: [`01_next_token_gru`](../parte-1-motor/01_next_token_gru.ipynb) · [`02_el_vaso_no_basta`](../parte-1-motor/02_el_vaso_no_basta.ipynb) · [`03_mini_transformer`](../parte-1-motor/03_mini_transformer.ipynb)

## Idea en una frase

Un modelo de lenguaje es una máquina que, dado el texto hasta aquí, reparte probabilidad entre los posibles siguientes tokens; entrenarlo es premiarlo cuando le dio mucha probabilidad al que de verdad venía.

---

## 1. Qué es un LLM

**Dummy.** Juegas a "adivina la siguiente letra" con el Quijote. Nadie te da respuestas: la respuesta es el propio libro, una letra más adelante. Juegas millones de veces y te vuelves muy bueno. Un LLM es eso, con tokens en vez de letras y con internet en vez del Quijote.

**Matemática.** Por la regla de la cadena, cualquier distribución sobre secuencias se factoriza como

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}).
$$

Un LM aprende $P_\theta(x_t\mid x_{<t})$ con una red de parámetros $\theta$. "Grande" se refiere al número de parámetros $N$ y de tokens de entrenamiento $D$, no a una arquitectura distinta.

**Lo que no es.** No es una base de datos que se consulta ni un buscador. No "sabe" la fecha. Genera la continuación probable de lo que le diste.

## 2. Vocabulario y ventanas: dónde está la etiqueta

**Dummy.** Cada carácter distinto del corpus recibe un número (`stoi`) y de vuelta (`itos`). Cortas el texto en ventanas de largo $L$. La entrada es la ventana; la etiqueta es **la misma ventana corrida un lugar**.

```
texto:   E n ␣ u n ␣ l u g a r
x:       E n ␣ u n ␣ l u g a
y:         n ␣ u n ␣ l u g a r
```

**Matemática.** Para una ventana que empieza en $i$: $x = (c_i,\dots,c_{i+L-1})$, $y=(c_{i+1},\dots,c_{i+L})$. Cada ventana da $L$ problemas de clasificación a la vez, uno por posición.

**Error típico.** Recorrer todas las ventanas posibles del Quijote (~2 M) en CPU. Los notebooks **muestrean** ventanas: misma distribución, mucho menos tiempo.

## 3. Entrenar: entropía cruzada y perplejidad

**Dummy.** Si al token correcto le diste probabilidad 0.9, pagas poquito. Si le diste 0.01, pagas mucho. Logaritmo: pagar por sorpresa.

**Matemática.**

$$
\mathcal{L}(\theta) = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\log P_\theta(y_{b,t}\mid x_{b,\le t}),
\qquad \mathrm{PPL} = e^{\mathcal{L}}.
$$

La perplejidad es "entre cuántas opciones igual de probables dudas, en promedio". **Ancla:** el Quijote completo tiene $|V|=91$ caracteres distintos. Un modelo que reparte uniforme tiene $\mathcal{L}=\ln 91\approx 4.51$ y $\mathrm{PPL}=91$. Si tu loss **antes de entrenar** no está cerca de 4.5, algo está mal.

**Train vs val.** Separa el final del texto (10%) como validación. Si train baja y val sube, memorizas.

## 4. GRU: todo el pasado en un vaso

**Dummy.** La GRU lee letra por letra y lleva un post-it $h_t$ de tamaño fijo con "lo que importa hasta ahora". Dos compuertas deciden cuánto borrar y cuánto escribir.

**Matemática.**

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1}) &&\text{(cuánto actualizar)}\\
r_t &= \sigma(W_r x_t + U_r h_{t-1}) &&\text{(cuánto del pasado usar)}\\
\tilde h_t &= \tanh\bigl(W x_t + U(r_t\odot h_{t-1})\bigr)\\
h_t &= (1-z_t)\odot h_{t-1} + z_t\odot \tilde h_t
\end{aligned}
$$

## 5. El vaso no basta

**Dummy.** Darle más letras atrás a la GRU es darle más tarea al mismo post-it. Si el post-it ya estaba lleno, más contexto no ayuda; a veces empeora.

**Matemática.** Todo lo que el modelo sabe del pasado pasa por $h_{t-1}\in\mathbb{R}^d$:

$$
I(x_{\le t-L};\, x_t \mid h_{t-1}) \le I(h_{t-1};\, x_t),
$$

y $h$ no crece con $L$. Además, el gradiente que llega a $h_{t-L}$ es un producto de $L$ jacobianos; si sus normas son $<1$ se desvanece. Esa es la motivación de la atención: en vez de un vaso, **mirar directo** a cada posición pasada.

## 6. Atención y el Transformer causal

**Dummy.** Cada posición hace una pregunta (query), cada posición pasada ofrece una etiqueta (key) y un contenido (value). Se mezclan los contenidos con pesos según qué tan bien contesta cada etiqueta a la pregunta.

**Matemática.**

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\qquad M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}
$$

- $\sqrt{d_k}$: sin esto, los productos punto crecen con la dimensión y el softmax se satura.
- $M$ (máscara causal): sin ella, la posición $i$ ve la respuesta $x_{i+1}$. La loss baja de forma espectacular y la generación es basura.
- **Posición**: la atención sola es invariante a permutaciones. Se suma un embedding de posición para que "perro muerde hombre" ≠ "hombre muerde perro".
- **Residual** $x + f(x)$ y LayerNorm: el gradiente tiene un camino directo hacia atrás.

**Costo.** $O(T^2 d)$ por capa: cada posición mira a todas las anteriores. Por eso la ventana de contexto es finita y cara.

## 7. Generar: temperatura

**Dummy.** El modelo da puntajes. La temperatura decide qué tan "atrevido" es al elegir: baja = siempre lo más probable (repetitivo), alta = cualquier cosa (incoherente).

**Matemática.**

$$
P_T(v) = \frac{\exp(z_v/T)}{\sum_u \exp(z_u/T)}.
$$

$T\to 0$: greedy (argmax). $T=1$: la distribución aprendida. $T>1$: más plana. *Top-p* recorta a los tokens que suman probabilidad $p$ antes de muestrear.

**El loop que se come a sí mismo.** Al generar, cada token elegido se vuelve entrada del siguiente paso. Un error temprano se arrastra.

---

## Autoevaluación

1. Señala en el código del notebook 01 la línea donde se construye la etiqueta. ¿Por qué no hace falta anotar datos a mano?
2. Tu loss antes de entrenar es 2.1 con un vocabulario de 91 caracteres. ¿Qué sospechas?
3. Alargas `seq_len` de 64 a 256 en la GRU y la val loss no baja. Da dos razones.
4. ¿Qué pasa con la loss de entrenamiento si quitas la máscara causal? ¿Y con las muestras?
5. ¿Por qué con $T=0.2$ el texto se repite?

<details>
<summary>Respuestas</summary>

1. En `CharWindowDataset.__getitem__`: `y = self.data[idx + 1 : idx + self.seq_len + 1]`, o sea `x` recorrido una posición. El texto es su propia etiqueta: aprendizaje autosupervisado.
2. Muy por debajo de $\ln 91\approx 4.51$ antes de entrenar: probablemente hay fuga (la etiqueta se coló en la entrada, p. ej. sin máscara o con un desfase mal hecho).
3. (a) El estado $h$ tiene la misma dimensión: más contexto no cabe. (b) El gradiente hacia posiciones lejanas se desvanece. (Bonus: con `max_windows` fijo, ventanas más largas = menos ventanas distintas.)
4. La loss baja mucho (el modelo copia el siguiente carácter, que ve). Las muestras son basura porque al generar ese futuro no existe.
5. Casi todo el peso se va al token más probable; los ciclos de alta probabilidad ("de la de la…") se vuelven atractores.

</details>
