# Reto · Tu oficio, no el nuestro

Construye un sistema con las tres piezas del módulo — **motor**, **lentes**, **reportero** — para un oficio que **no** sea el noticiero copiado. Demo de 8 minutos.

PDF histórico del enunciado: [`pdfs/05_02_RetoLLM.pdf`](../pdfs/05_02_RetoLLM.pdf). Lo que manda es este README y el capítulo 5 del ebook.

## Lo que tienes que poder decir en la demo

- Un LLM apuesta el siguiente token.
- Yo no entrené el cerebro; le puse lentes (LoRA).
- Los datos del día no están en los lentes; los trae el reportero.
- Los lentes enseñan el oficio y a no inventar.
- Si le quito el papel, vuelve a alucinar.

## Ideas de oficio

- Briefing de movilidad CDMX (alertas, no "el tráfico en general").
- "Explícame este DOF / esta Gaceta como si tuviera 15 años."
- Guardia de tu materia: preguntas de tu tesis con papers pegados.
- Resumen de 3 papers de tu lab, sin citar lo que no está en el PDF.
- Noticiero de un municipio, un deporte o un mercado, con feed propio.

## Requisitos mínimos

| Pieza | Qué entregar | Material de apoyo |
|---|---|---|
| **Motor** | Lab 0 o mini-GPT: curvas train/val, 3 temperaturas, y en el código la flecha "aquí está la etiqueta". Corpus documentado. | `parte-1-motor/01`, `03` |
| **Lentes** | Base open-weight ≤ 1.7 B (Qwen2.5-0.5B, SmolLM2-360M/1.7B, Llama-3.2-1B…). LoRA o QLoRA. Adapter + `adapter_config.json` + `entrenamiento.json` (`r`, `lr`, épocas, semilla, `model_id`). | `parte-3-oficio/07_gym_lora.ipynb` |
| **Dataset** | JSONL de chat propio, **≥ 100** filas revisadas a mano. Regla de oro aplicada: cuántas tiraste y por qué. | `parte-3-oficio/06`, `don_titular.py` |
| **Reportero** | Script que trae los datos frescos. Documenta origen, fecha y qué pasa si la fuente está caída. Sin papel, el sistema se niega o avisa — no inventa. | `parte-3-oficio/rss_reportero.py` |
| **Juicio** | Tabla antes vs después en ≥ 5 prompts de hold-out separados por fecha: `zero-shot`, `few-shot`, `lora`, más tu columna humana. | `parte-3-oficio/08` |

## Demo de 8 minutos

1. Tu oficio y por qué te importa (30 s).
2. Diagrama: motor / lentes / reportero, señalando dónde está el día, el estilo y el idioma.
3. Tabla antes vs después en hold-out.
4. En vivo: entra un dato fresco, sale el oficio.
5. En vivo: le quitas el papel. ¿Se niega o inventa?
6. Una limitación honesta y una mejora concreta (cómputo, datos o eval). "Faltó tiempo" no cuenta.

## Checklist de entrega

- [ ] Lab 0 o mini-GPT: curvas + 3 temperaturas + "la etiqueta está aquí".
- [ ] `model_id` ≤ 1.7 B + adapter + `adapter_config.json` + semilla.
- [ ] JSONL propio ≥ 100 filas; `python don_titular.py mi-dataset.jsonl` sin errores (o tu juez adaptado a tu formato).
- [ ] Conteo de filas tiradas (por el juez y a mano), con 3 ejemplos.
- [ ] Reportero documentado; la inferencia sin papel se niega o avisa.
- [ ] Tabla antes/después en hold-out; tu LoRA comparado contra **few-shot**, no solo contra zero-shot.
- [ ] Diagrama motor / lentes / reportero.
- [ ] Una limitación que no sea "faltó tiempo".

## Extensiones (si te sobra)

- Comparar dos bases (0.5 B vs 1.7 B) con el mismo dataset.
- QLoRA vs LoRA: VRAM, tiempo y calidad del juicio.
- Una regla nueva en el juez (atribución, negaciones) y cuántos casos nuevos caza.
- Servir en Ollama / GGUF, no solo Gradio.
- Un segundo oficio con el mismo adapter: ¿dónde se rompe?

## Si tu oficio no es "5 bullets"

`don_titular.py` está hecho para el noticiero. Cópialo y cambia `SYSTEM`, `N_BULLETS` y `parse_briefing` a tu formato. Lo que no cambias es la idea: **cada hecho de la salida tiene que poder señalarse en la entrada**.
