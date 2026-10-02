# Notas de estudio

Resumen para repasar **después** de correr cada notebook. El ebook explica con calma; estas notas son lo que tienes que poder decir en voz alta en el examen o en la demo del reto.

| Nota | Notebooks | En una frase |
|---|---|---|
| [`01_el_motor.md`](01_el_motor.md) | 01 · 02 · 03 | Un LLM apuesta el siguiente token; la etiqueta es el texto recorrido un lugar. |
| [`02_el_chef.md`](02_el_chef.md) | 04 · 05 | El modelo real ya existe; se le habla con su tokenizer y su chat template, y sin notas inventa el día. |
| [`03_el_oficio.md`](03_el_oficio.md) | 06 · 07 · `sft_lora_noticias.py` | El día lo trae el reportero, el oficio lo enseñan los lentes (LoRA), y un juez lo comprueba. |
| [`formulas.md`](formulas.md) | todos | Las 12 fórmulas del módulo en una hoja. |
| [`glosario.md`](glosario.md) | todos | Términos en español ↔ inglés, con la definición que usamos aquí. |

## Cómo estudiar con esto

1. Corre el notebook. No leas la nota antes: primero ensúciate las manos.
2. Lee la nota y tapa la sección **Matemática**. ¿Puedes reconstruir la fórmula desde la sección **Dummy**?
3. Contesta la **autoevaluación** sin abrir las respuestas. Si fallas una, vuelve al notebook, no a la nota.

## El hilo de todo el módulo

```
Parte I   motor      P(x_t | x_<t)  lo construyes tú: letras, GRU, atención
Parte II  chef       el mismo P, pero con 494 M de parámetros que no entrenaste
Parte III oficio     P(y | system, notas): notas del reportero + lentes LoRA + juez
```

Si en la demo del reto alguien pregunta "¿y dónde está la inteligencia?", la respuesta tiene tres partes: el **idioma** está en el pretrain (no lo hiciste tú), el **día** está en las notas (no está en el modelo) y el **oficio** está en el adapter (eso sí lo entrenaste tú).
