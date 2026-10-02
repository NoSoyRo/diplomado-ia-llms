# Glosario

Definiciones como las usamos en el módulo. Entre paréntesis, el término en inglés que vas a ver en papers y librerías.

| Término | Definición |
|---|---|
| **Adapter** | Archivo con los pesos LoRA ($A$, $B$). Se carga encima del modelo base. Pesa MB, no GB. |
| **Alucinación** (*hallucination*) | Texto con forma de hecho que no tiene soporte en la entrada ni en la realidad. En el módulo: cualquier hecho del briefing que no está en las notas. |
| **Atención** (*attention*) | Mezcla de valores de posiciones pasadas, con pesos según el parecido query–key. |
| **Base** (*base model*) | Modelo solo preentrenado: continúa texto. |
| **BPE** (*Byte Pair Encoding*) | Algoritmo que construye el vocabulario fusionando la pareja de símbolos más frecuente, repetidamente. |
| **Byte-level** | Tokenizer cuya base son los 256 bytes: cualquier texto tiene representación. |
| **Chat template** | Función $\varphi$ que convierte la lista de mensajes en el string con marcas especiales que el modelo leyó en su ajuste. |
| **Contexto / ventana** (*context window*) | Número máximo de tokens que el modelo ve a la vez. |
| **Entropía cruzada** (*cross-entropy*) | $-\log$ de la probabilidad asignada al token correcto, promediada. |
| **Few-shot** | Poner ejemplos resueltos en el prompt. Aprendizaje sin gradiente (*in-context learning*). |
| **Greedy** | Generar eligiendo siempre el token más probable ($T\to 0$). |
| **Hold-out** | Datos que no se usan para entrenar ni para elegir hiperparámetros; solo para la evaluación final. |
| **Instruct** | Modelo base con ajuste adicional sobre diálogos: sigue instrucciones dentro de su chat template. |
| **Juez** | Aquí: `don_titular.py`. Reglas reproducibles (formato, citas, soporte) que califican un briefing sin IA. |
| **LoRA** (*Low-Rank Adaptation*) | Ajuste que congela $W$ y entrena un parche de rango bajo $\frac{\alpha}{r}BA$. |
| **Máscara causal** | Matriz $M$ que impide que una posición vea las siguientes. |
| **Máscara de pérdida** ($m_t$) | En SFT, 1 en los tokens del assistant y 0 en el resto: solo se aprende a escribir la respuesta. |
| **Merge** | Un paso de BPE: la pareja $(a,b)$ que se vuelve el símbolo $ab$. |
| **Open-weight** | Modelo cuyos pesos puedes descargar (Qwen, Llama, SmolLM). |
| **Perplejidad** (*perplexity*) | $e^{\mathcal{L}}$: número efectivo de opciones entre las que duda el modelo. |
| **Pretrain** | Entrenamiento next-token sobre un corpus enorme. De ahí sale el idioma. |
| **QLoRA** | LoRA con el modelo base cuantizado a 4 bits. |
| **RAG** (*Retrieval-Augmented Generation*) | Traer documentos $d$ y generar condicionado a ellos: $P(y\mid q,d)$. |
| **Regla de oro** | Si el assistant dice algo que no está en el user, la fila se tira. |
| **Reportero** | `rss_reportero.py`: el retriever del noticiero. Cero IA. |
| **SFT** (*Supervised Fine-Tuning*) | Ajuste con pares (prompt, respuesta) y cross-entropy solo sobre la respuesta. |
| **Soporte** | Fracción de las raíces de un bullet que aparecen en las notas. |
| **Temperatura** | Divide los logits antes del softmax: baja = conservador, alta = disperso. |
| **Token** | Unidad que el modelo ve: un carácter (Parte I) o un pedazo de BPE (Parte II en adelante). |
| **Top-p** (*nucleus sampling*) | Muestrear solo entre los tokens más probables que suman probabilidad $p$. |
| **Villano** | La pregunta "¿qué pasó hoy?" sin notas: el modelo inventa con seguridad. |
