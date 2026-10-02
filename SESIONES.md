# Sesiones · 8 actividades

Una sesión, un notebook, una actividad. Cada actividad deja un **entregable corto** (capturas + 2–3 frases) que sirve de práctica; lo que se califica es el [reto](reto/README.md): código + `reporte.pdf` del mini-LLM.

Las sesiones 1–3 construyen el reto. Las 4–8 muestran lo mismo en un LLM de verdad y alimentan la sección 8 del reporte (limitaciones y mejora).

| # | Notebook | Actividad | Entregable | Para el reto |
|---|---|---|---|---|
| 1 | [`01_next_token_gru`](parte-1-motor/01_next_token_gru.ipynb) | ¿Quién pone la etiqueta? | tabla train/val/ppl, 3 temperaturas, la línea donde nace $y$ | requisitos 3.1–3.6 |
| 2 | [`02_el_vaso_no_basta`](parte-1-motor/02_el_vaso_no_basta.ipynb) | El experimento justo | tabla ingenua vs justa, curva por posición | extensión `seq_len` |
| 3 | [`03_mini_transformer`](parte-1-motor/03_mini_transformer.ipynb) | Quita la máscara | loss y muestra con y sin máscara | extensión GRU vs Transformer |
| 4 | [`04_bpe_a_mano`](parte-2-chef/04_bpe_a_mano.ipynb) | Tu nombre en tokens | tabla merges vs letras por token; tu nombre en BPE y en Qwen | sección 8: por qué carácter es caro |
| 5 | [`05_chat_template_y_villano`](parte-2-chef/05_chat_template_y_villano.ipynb) | Provoca al villano | 3 salidas con el juez; una frase inventada que el juez no vio | sección 8: inventa con confianza |
| 6 | [`06_reportero_y_dataset`](parte-3-oficio/06_reportero_y_dataset.ipynb) | La caza de tramposas | 20 filas de tu oficio; cuántas tramposas cazó el juez y cuántas tú | sección 2: limpiar datos |
| 7 | [`07_gym_lora`](parte-3-oficio/07_gym_lora.ipynb) | El gym | curva train/val, `entrenamiento.json`, con vs sin lentes | sección 6: memorizó o generalizó |
| 8 | [`08_juicio_antes_despues`](parte-3-oficio/08_juicio_antes_despues.ipynb) | La caza del Goodhart | tabla de 3 condiciones; un caso con 0 problemas que está mal | sección 8: medir bien |

---

## 1 · ¿Quién pone la etiqueta?

- **Antes de correr (5 min).** Cada quien escribe qué pérdida espera en el paso 0. Se revela el ancla: $\ln 91\approx 4.51$.
- **Corren (15 min).** Comparan su perplejidad final contra 91.
- **En voz alta.** Uno pasa a señalar en el código dónde nace $y$. Si dice "alguien la etiquetó", se repite.
- **Entregable.** Tabla época/train/val/ppl, las 3 muestras, la línea de $y$.
- **Tarea para el reto.** Elegir corpus (10–500 KB, no el Quijote) y traerlo a la sesión 2.

## 2 · El experimento justo

- **Votación (5 min).** "¿L=256 le gana a L=32?" Se anota el resultado de la votación.
- **Versión ingenua (10 min).** `igualar_presupuesto = False` y vuelven a correr: 4 000 ventanas y batch fijo. L=256 lee 8× más letras y "gana".
- **Versión justa (10 min).** `igualar_presupuesto = True` (lo que ya trae el notebook). Mismas letras, mismos pasos: empatan. La curva por posición se aplana en ~30 letras.
- **Discusión.** ¿Qué cambió entre las dos? Un experimento justo cambia una sola cosa.
- **Entregable.** Las dos tablas y una frase sobre la trampa.
- **Tarea para el reto.** Correr el 01 con **su** corpus.

## 3 · Quita la máscara

- **Corren (15 min).** El 03 tal cual. Comparan con la GRU de la sesión 1 (el Transformer pierde en CPU: ¿por qué?).
- **Rompen (15 min).** Comentan `masked_fill`. La loss se desploma y la muestra es basura. Explican por qué en dos frases.
- **Entregable.** Loss y muestra con y sin máscara.
- **Tarea para el reto.** Elegir GRU o Transformer y fijar `config.json`.

## 4 · Tu nombre en tokens

- **Corren (20 min).** BPE desde cero sobre el Quijote.
- **Juego (10 min).** Tokenizan su nombre y una palabra de 2026 con `codificar` y con el tokenizer de Qwen (sección 6 del notebook). ¿Quién tiene el nombre más caro?
- **Entregable.** La tabla merges / letras por token / letras en una ventana de 64; su nombre en los dos tokenizers.
- **Para el reto.** Una frase para la sección 8: con 1 000 merges la misma ventana ve ~3× más texto; por carácter, tu modelo ve 64 letras y nada más.

## 5 · Provoca al villano

- **Corren (15 min).** Qwen base vs Instruct, el chat template.
- **Concurso (15 min).** Cada quien busca el prompt que más lo haga inventar. El juez califica; se exhibe el peor de la clase.
- **Entregable.** Villano, sin notas, con notas, cada uno con su `resumen()`; una frase inventada que el juez no vio.
- **Para el reto.** Sección 8: tu mini-LLM y Qwen inventan por la misma razón (apuestan el siguiente token).

## 6 · La caza de tramposas

- **Escriben (20 min).** 20 filas de **su** oficio con `agregar`.
- **En parejas (20 min).** Cada quien le mete 2 filas tramposas al otro (número cambiado, "dice lo contrario"). El otro las caza con el juez y a mano.
- **Entregable.** Las 20 filas; cuántas tramposas cazó el juez, cuántas tú.
- **Para el reto.** Sección 2: qué limpiaste en tu corpus y por qué.

## 7 · El gym

- **Entrenan (20 min, Colab T4).** Con el dataset de ejemplo. Leen la curva: train cruza a val hacia la época 4.
- **Rompen (15 min).** `epochs = 2` y `r = 4`. ¿Qué aprende primero, el formato o las citas?
- **Entregable.** Curva, `entrenamiento.json`, una salida con y sin lentes.
- **Para el reto.** Sección 6: la misma pregunta (¿memorizó o generalizó?) con tu curva.

## 8 · La caza del Goodhart

- **Corren (10 min).** La tabla zero-shot / few-shot / LoRA en 10 días de hold-out.
- **Cazan (25 min).** Una salida con **0 problemas** del juez que esté mal (pista: h10). Luego agregan una regla al juez y cuentan cuántos casos nuevos caza.
- **Entregable.** El caso, la regla y el conteo.
- **Para el reto.** Sección 8: un número bajo no es un modelo bueno; di qué no mide tu perplejidad.

---

**Exit ticket de cada sesión:** la autoevaluación de la nota que corresponde ([`notas/`](notas/)), 5 preguntas, 5 minutos.

**Entrega del reto:** después de la sesión 8. Rúbrica en [`reto/README.md`](reto/README.md).
