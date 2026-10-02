# Hoja de fórmulas

| # | Qué | Fórmula | Dónde |
|---|---|---|---|
| 1 | Regla de la cadena (qué es un LM) | $P(x_{1:T})=\prod_t P_\theta(x_t\mid x_{<t})$ | 01 |
| 2 | Entropía cruzada | $\mathcal{L}=-\frac{1}{BT}\sum_{b,t}\log P_\theta(y_{b,t}\mid x_{b,\le t})$ | 01 |
| 3 | Perplejidad | $\mathrm{PPL}=e^{\mathcal{L}}$; uniforme sobre $\lvert V\rvert$ ⇒ $\mathcal{L}=\ln\lvert V\rvert$ | 01 |
| 4 | Temperatura | $P_T(v)=\dfrac{e^{z_v/T}}{\sum_u e^{z_u/T}}$ | 01, 05 |
| 5 | Cuello de botella del estado | $I(x_{\le t-L};x_t\mid h_{t-1})\le I(h_{t-1};x_t)$, $h\in\mathbb{R}^d$ fijo | 02 |
| 6 | Atención causal | $\mathrm{softmax}\!\left(QK^\top/\sqrt{d_k}+M\right)V$, $M_{ij}=-\infty$ si $j>i$ | 03 |
| 7 | Merge de BPE | $(a,b)^\star=\arg\max_{(a,b)}\sum_w c(w)\,\#(a,b \text{ en } w)$ | 04 |
| 8 | Texto por ventana | chars vistos $\approx T\cdot\bar\ell$ (chars/token) | 04 |
| 9 | RAG casero | $\hat y\sim P_\theta(\cdot\mid\varphi(s,q,d))$, $d=R(q)$ | 05, 06 |
| 10 | LoRA | $W'=W+\frac{\alpha}{r}BA$; params $r(d+k)$ vs $dk$ | `sft_lora_noticias.py` |
| 11 | SFT enmascarado | $\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{\sum m_t}\sum_t m_t\log P_\theta(x_t\mid x_{<t})$ | 06 |
| 12 | Soporte del juez | $\lvert S(b)\cap S(d)\rvert/\lvert S(b)\rvert\ge 0.5$ y $N(b)\subseteq N(d)$ | 06, 07 |

**Extras (Parte I, `extra/`):**

- Búsqueda de hiperparámetros: $\lambda^\star\in\arg\min_\lambda \mathcal{L}_{\text{val}}(\theta^\star(\lambda))$.
- Scaling (forma de Kaplan, ilustrativa): $\mathcal{L}(N,D)\approx E_\infty+(N_c/N)^{\alpha}+(D_c/D)^{\beta}$.
- Batch efectivo: $B_{\text{eff}}=B_{\text{micro}}\times G$.

## Números que conviene saber de memoria

| Qué | Valor |
|---|---|
| $\ln 91$ (loss de un modelo uniforme sobre los 91 caracteres del Quijote) | ≈ 4.51 |
| Chars/token: caracteres · BPE 1 000 merges · Qwen (Quijote) | 1.00 · 2.83 · 3.20 |
| Qwen2.5-0.5B: parámetros · capas · $d$ · vocabulario | 494 M · 24 · 896 · 151 665 |
| LoRA r=16 en `q_proj`+`v_proj` de Qwen2.5-0.5B | 1.08 M (0.22%) |
