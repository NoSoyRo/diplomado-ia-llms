# Reto · Tu mini-LLM

Diseña, entrena y evalúa un modelo de lenguaje **por carácter** (GRU o mini-Transformer causal) sobre un texto que tú elijas. Se califica con dos cosas: **tu código** y **un `reporte.pdf` que explica todo**.

Enunciado oficial: [`pdfs/05_02_RetoLLM.pdf`](../pdfs/05_02_RetoLLM.pdf). Este README es ese enunciado en checklist, más cómo se califica.

Material de apoyo: [`parte-1-motor/01`](../parte-1-motor/01_next_token_gru.ipynb) (GRU), [`02`](../parte-1-motor/02_el_vaso_no_basta.ipynb) (contexto), [`03`](../parte-1-motor/03_mini_transformer.ipynb) (Transformer) y [`notas/01_el_motor.md`](../notas/01_el_motor.md). Puedes partir de ese código; lo que no puedes es entregarlo tal cual sobre el Quijote.

---

## Qué entregas

Un repo (o carpeta comprimida) con:

```
mi-mini-llm/
├── corpus.txt          tu texto (10 KB – 500 KB), NO el Quijote
├── entrenar.py         o un notebook: corre de cero con un comando
├── config.json         hiperparámetros, arquitectura, rutas y semilla de la corrida del reporte
├── resultados/         log de train/val por época, la figura de la curva, las 3 muestras
└── reporte.pdf         lo que se califica junto con el código
```

## Requisitos (del enunciado oficial)

| # | Requisito | En el código | En el PDF |
|---|---|---|---|
| 3.1 | **Datos**: un `.txt` de 10–500 KB, origen documentado, limpieza mínima si hace falta | carga + limpieza en una función | de dónde salió, tamaño en KB, qué limpiaste y por qué |
| 3.2 | **Tokenización por carácter**: vocabulario, `stoi`, `itos` explícitos | `build_vocab`, `encode`, `decode` | $\lvert V\rvert$ y 5 caracteres "raros" de tu corpus |
| 3.3 | **Modelo**: GRU **o** mini-Transformer causal; next-token; sin ver el futuro; distribución sobre $V$ | la clase del modelo; máscara causal si es Transformer | diagrama o tabla de capas, número de parámetros, por qué esa arquitectura |
| 3.4 | **Entrenamiento**: entropía cruzada, Adam/AdamW, mini-batches; registra train_loss, val_loss y épocas | loop de entrenamiento + log | la curva train/val y qué te dice |
| 3.5 | **Evaluación**: val_loss final y perplejidad = `exp(val_loss)` | se imprime y se guarda | comparada contra el ancla $\ln\lvert V\rvert$ (modelo que no aprendió nada) |
| 3.6 | **Generación**: prompt fijo, ≥ 3 temperaturas, 300–600 tokens | función `generate` con temperatura | las 3 muestras y qué cambia con $T$ |
| 3.7 | **Reproducibilidad**: semillas fijas, configuración completa guardada | `torch.manual_seed`, `config.json` | la tabla de configuración tal cual |

## Qué tiene que explicar el `reporte.pdf`

"Explicar todo" quiere decir que alguien que no vio tu código entiende qué hiciste y por qué. En este orden:

1. **Portada.** Nombre completo, número de cuenta o ID del diplomado, fecha.
2. **Datos.** Origen, tamaño, limpieza, $\lvert V\rvert$.
3. **El problema.** Qué es next-token prediction. La regla de la cadena $P(x_1,\dots,x_T)=\prod_t P(x_t\mid x_{<t})$ y la entropía cruzada que minimizas. **Pega las líneas de tu código donde nace la etiqueta $y$** y explica por qué nadie la anotó a mano.
4. **El modelo.** Arquitectura, tamaños, número de parámetros. Si es Transformer: qué hace la máscara causal y qué pasa sin ella. Si es GRU: qué es el estado $h$ y su límite.
5. **Entrenamiento.** Hiperparámetros (la tabla de `config.json`), cómo separaste train/val y por qué no hay fuga, la curva.
6. **Resultados.** val_loss final, perplejidad, el ancla $\ln\lvert V\rvert$, y una frase: ¿memorizó o generalizó? (mira la distancia entre train y val).
7. **Generación.** Las 3 muestras con el mismo prompt. Qué aprendió (ortografía, puntuación, palabras de tu corpus) y qué no (sentido, frases largas).
8. **Limitaciones y una mejora.** Una limitación medida (no "faltó tiempo") y una mejora concreta en datos, cómputo o arquitectura, con el número que esperas mover.

Extensiones opcionales del enunciado (GRU vs Transformer, efecto de `seq_len`, tamaño del modelo, acumulación de gradientes, AMP, estudio de hiperparámetros): suman en el PDF si las mides **con un experimento justo** (cambia una sola cosa; ver [notebook 02](../parte-1-motor/02_el_vaso_no_basta.ipynb)). No sustituyen ningún requisito.

---

## Cómo se califica

**Nota = 50% código + 50% reporte.** Cada eje se califica sobre 10.

### Código (50%)

| Criterio | Puntos | Qué se revisa |
|---|---|---|
| Datos y tokenización | 2 | carga, limpieza, `stoi`/`itos` explícitos; `decode(encode(s)) == s` |
| Modelo | 2 | GRU o Transformer escrito por ti; logits de tamaño $\lvert V\rvert$; causal |
| Entrenamiento | 2 | entropía cruzada, Adam/AdamW, mini-batches, train/val sin fuga, log por época |
| Evaluación y generación | 2 | val_loss y perplejidad; `generate` auto-regresivo con temperatura; ≥ 3 temperaturas |
| Reproducibilidad | 2 | semilla, `config.json`, corre de cero con un comando y da los números del reporte |

**Techos del código:**

- No corre con el comando que dice el reporte → ≤ 5
- Tokenización que no es por carácter → ≤ 6
- Transformer sin máscara causal (o GRU bidireccional) → ≤ 5
- Sin validación (solo train_loss) → ≤ 6
- El notebook del curso tal cual sobre el Quijote → ≤ 4

### Reporte (50%)

| Criterio | Puntos | Qué se revisa |
|---|---|---|
| Datos y problema | 2 | secciones 2–3; la etiqueta señalada en **tu** código |
| Modelo y entrenamiento | 2 | secciones 4–5; decisiones justificadas, no solo listadas |
| Resultados | 2 | sección 6; curva, perplejidad contra el ancla, memorizó/generalizó |
| Generación | 2 | sección 7; 3 muestras y lectura de cada temperatura |
| Limitaciones y mejora | 2 | sección 8; medida y concreta |

**Techos del reporte:**

- Sin `reporte.pdf`, o PDF que es imagen (texto no seleccionable) → ≤ 5
- Portada sin nombre o sin número de cuenta / ID → ≤ 6
- Números del PDF que no salen de tu código o `resultados/` → ≤ 5
- Párrafos genéricos (de un LLM o del ebook) sin tus números → ≤ 6
- "Faltó tiempo" como limitación → la sección 8 vale 0

Extra (LaTeX, figuras de más, una sim, un segundo modelo) **no resta**.

**Revisión automática (evidencia, no la nota):**

```bash
pip install -r ../requirements-dev.txt   # trae pypdf
python ../scripts/revisar_reto.py <carpeta_de_tu_entrega>
```

Corre los checks verificables (estructura, tamaño del corpus, tokenización
por carácter, arquitectura, entropía cruzada, semilla, texto extraíble del
PDF, cruce perplejidad-vs-`resultados/`). Quien califique usa
[`AGENTE_reto_llms.md`](AGENTE_reto_llms.md) para combinar esa evidencia con
la rúbrica de abajo.

## Checklist antes de entregar

- [ ] `corpus.txt` de 10–500 KB, que no es el Quijote, con origen en el PDF.
- [ ] `stoi`, `itos`, `encode`, `decode` explícitos.
- [ ] GRU o Transformer causal; next-token; logits sobre $V$.
- [ ] Entropía cruzada + Adam/AdamW + mini-batches; train/val por época guardados.
- [ ] val_loss final + perplejidad + ancla $\ln\lvert V\rvert$.
- [ ] Prompt fijo, 3 temperaturas, 300–600 tokens.
- [ ] Semilla + `config.json`; corre de cero con un comando.
- [ ] `reporte.pdf` con texto seleccionable, portada completa y las 8 secciones.
- [ ] En el PDF, las líneas de tu código donde nace $y$.

## Lo que tienes que poder decir en el oral (30 s cada una)

- ¿Dónde está la etiqueta y por qué nadie la anotó?
- ¿Qué significa tu perplejidad? ¿Contra qué la comparas?
- ¿Qué pasa con la loss si quitas la máscara causal (o si $y$ fuera igual a $x$)?
- ¿Por qué con $T$ baja se repite y con $T$ alta se rompe?
- ¿Tu modelo memorizó o generalizó? ¿Qué número lo dice?

---

## El resto del módulo (no se califica en el reto)

Las Partes II y III (Qwen, chat template, LoRA, reportero, juez) son las **actividades** de las sesiones 4–8: ver [`SESIONES.md`](../SESIONES.md). Sirven para la sección 8 del reporte: un LLM de verdad tiene los mismos tres problemas que tu mini-LLM (no sabe el día, inventa, el formato hay que enseñárselo), a otra escala.
