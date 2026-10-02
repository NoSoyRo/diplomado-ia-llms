# Plantilla de la entrega

Esto **no** es el reto. Son los nombres de archivo que `scripts/revisar_reto.py` sabe leer. Cópialos a tu carpeta y reemplaza los ceros.

```
mi-mini-llm/
├── corpus.txt
├── entrenar.py          (o un .ipynb)
├── config.json          ← parte de config.ejemplo.json
├── resultados/
│   ├── historia.json    ← {"train_loss": [...], "val_loss": [...]} por época
│   ├── curva.png
│   └── muestras.txt     las 3 temperaturas, una debajo de otra
└── reporte.pdf
```

`historia.json` tiene que llamarse así (o `historia-algo.json`). `ESQUEMA_historia.json` de esta carpeta **no** cuenta: el revisor busca `historia*.json` dentro de `resultados/`.

La perplejidad del PDF tiene que ser `exp(val_loss[-1])` de ese archivo. Si no cuadra, la revisión lo marca.
