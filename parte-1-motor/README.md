# Parte I · El motor (de cero)

El examen del módulo es esto: next-token a mano. Si no puedes señalar **dónde está la etiqueta** en el código, Qwen sigue siendo magia.

Notas de estudio: [`notas/01_el_motor.md`](../notas/01_el_motor.md). Corpus: [`datos/quijote.txt`](../datos/quijote.txt) (los notebooks lo encuentran solos; en Colab descomenta el `wget`).

## `01_next_token_gru.ipynb` — Lab 0

**Dummy.** Lees en voz alta y, en cada letra, apuestas qué sigue. Ganas si le diste alta probabilidad a la letra que *sí* venía. Nadie etiquetó nada a mano: la respuesta es el texto, recorrido un lugar.

**Math.** Un LM es la regla de la cadena $P(x_{1:T})=\prod_t P_\theta(x_t\mid x_{<t})$. Entrenar es minimizar la entropía cruzada $\mathcal{L}=-\frac{1}{BT}\sum\log P_\theta(y_{b,t}\mid x_{b,\le t})$. La GRU comprime todo el pasado en un vaso $h_t\in\mathbb{R}^{d}$ de capacidad fija.

**Entregable.** Curva train/val, muestras con 3 temperaturas, y una flecha en el código: *aquí está la etiqueta* (`y = x` recorrido uno).

## `02_el_vaso_no_basta.ipynb` — Más contexto no siempre ayuda

**Dummy.** "Más letras atrás ⇒ predice mejor" suena lógico. En una GRU a menudo es falso: le das más tarea al mismo post-it, no un post-it más grande.

**Math.** $I(x_{\le t-L};x_t\mid h_{t-1})\le I(h_{t-1};x_t)$ y $h$ tiene dimensión fija. Alargar $L$ no agranda $d$. Si $\partial\mathcal{L}/\partial h_{t-L}$ se desvanece, el vaso ni usa lo que cabría.

**Entregable.** Tabla `seq_len → val loss` y una frase: por qué no baja como esperabas.

## `03_mini_transformer.ipynb` — Un bloque decoder

**Dummy.** Cada letra voltea a ver el pasado con pesos ("de lo que ya leí, ¿qué me importa *ahora*?"). Sin máscara causal haces trampa: miras la respuesta. Sin posición, "perro muerde hombre" = "hombre muerde perro".

**Math.** $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k}+M)V$, con $M_{ij}=-\infty$ si $j>i$. Residual $x+f(x)$ para que el gradiente no se muera.

**Entregable.** Misma pregunta que el 01, otra máquina: compara la val loss y las muestras. Quita la máscara y explica por qué la loss "mejora" y la generación empeora.

---

## `extra/` — temario viejo, opcional

No son el eje. Sirven si quieres repetir ingeniería con el juguete.

| Archivo | Dummy | Math |
|---|---|---|
| `E1_grid_hiperparametros.ipynb` | El LR no se adivina; se mide. El grid es el experimento más tonto y más honesto. | $\lambda^\star\in\arg\min_{\lambda}\mathcal{L}_{\mathrm{val}}(\theta^\star(\lambda))$ |
| `E2_scaling.ipynb` | Más cerebro o más libro bajan la pérdida. No preentrenas GPT-4: compras el pan. | $\mathcal{L}(N,D)\approx E_\infty+(N_c/N)^\alpha+(D_c/D)^\beta$ |
| `E3_memoria_vram.ipynb` | El T4 no aguanta el batch del paper. Acumulas, mezclas precisión, recomputeas activaciones. | $B_{\mathrm{eff}}=B_{\mathrm{micro}}\times G$ |
