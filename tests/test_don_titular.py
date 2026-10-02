import json
from pathlib import Path

import don_titular as dt

RAIZ = Path(__file__).resolve().parents[1]
DATASET = RAIZ / "parte-3-oficio" / "ejemplo-briefing.jsonl"

NOTAS = """Notas del 2026-08-14:
1. El País — Banxico mantiene la tasa de referencia.
   El banco central dejó sin cambio el objetivo ante la inflación subyacente.
2. Local — Paro parcial en la L12 del Metro por fallas eléctricas.
   Autoridades estiman reanudar el servicio en la tarde.
3. BBC Mundo — Ola de calor en el norte de México.
   Varios estados superan los 40 °C.
"""

BUENO = """1) Banxico (El País): mantiene la tasa; menciona inflación subyacente.
2) Metro L12 (local): paro parcial por fallas eléctricas; reanudación estimada por la tarde.
3) Clima (BBC Mundo): ola de calor en el norte, varios estados >40 °C.
4) Política nacional: no aparece.
5) Deportes / mercados: no aparece."""


def test_parse_notas_separa_medio_titulo_y_cuerpo():
    notas = dt.parse_notas(NOTAS)
    assert [n.medio for n in notas] == ["El País", "Local", "BBC Mundo"]
    assert notas[2].cuerpo == "Varios estados superan los 40 °C."


def test_parse_notas_sin_medio_ignora_fuente():
    crudo = "Notas:\n1. Titular sin medio\n   cuerpo\n   fuente: https://x.y/z\n"
    (nota,) = dt.parse_notas(crudo)
    assert nota.medio == "" and nota.cuerpo == "cuerpo"


def test_briefing_canonico_aprueba():
    j = dt.juzgar(NOTAS, BUENO)
    assert j.aprobado, j.problemas
    assert j.formato_ok and j.citas_ok and j.soporte == 1.0
    assert j.con_hechos == 3


def test_numero_inventado_se_detecta():
    malo = BUENO.replace("mantiene la tasa", "sube la tasa 50 puntos")
    assert any("números" in p and "50" in p for p in dt.juzgar(NOTAS, malo).problemas)


def test_hecho_sin_soporte_se_detecta():
    malo = BUENO.replace("4) Política nacional: no aparece.", "4) Sismo (El País): terremoto en Oaxaca.")
    problemas = dt.juzgar(NOTAS, malo).problemas
    assert any(p.startswith("bullet 4: soporte") for p in problemas)


def test_medio_que_no_esta_en_notas():
    malo = BUENO.replace("(BBC Mundo)", "(Reforma)")
    assert any("Reforma" in p for p in dt.juzgar(NOTAS, malo).problemas)


def test_bullet_sin_cita():
    malo = BUENO.replace("Banxico (El País):", "Banxico:")
    assert "bullet 1: no cita medio" in dt.juzgar(NOTAS, malo).problemas


def test_formato_cuatro_bullets():
    malo = "\n".join(BUENO.splitlines()[:4])
    j = dt.juzgar(NOTAS, malo)
    assert not j.formato_ok


def test_no_aparece_no_cuenta_como_hecho():
    todo_vacio = "\n".join(f"{i}) Tema {i}: no aparece." for i in range(1, 6))
    j = dt.juzgar(NOTAS, todo_vacio)
    assert j.aprobado and j.con_hechos == 0 and j.soporte is None


def test_texto_sin_formato_se_juzga_por_renglones():
    salida = "El País - Banxico mantiene la tasa.\n\nBBC Mundo - Ola de calor en el norte."
    j = dt.juzgar(NOTAS, salida)
    assert not j.formato_ok and j.citas_ok
    assert j.con_hechos == 2 and j.soporte == 1.0


def test_renglon_suelto_inventado_sin_notas():
    salida = "1. La Compañía Federal de Trabajo (CFDTEC) anuncia medidas.\n2. Sismo en Oaxaca."
    j = dt.juzgar("", salida)
    assert j.con_hechos == 2 and j.soporte == 0.0 and not j.citas_ok


def test_frase_añadida_baja_el_soporte():
    salida = "El País - La tasa de referencia se mantuvo estable durante el año pasado."
    assert any("soporte" in p for p in dt.juzgar(NOTAS, salida).problemas)


def test_normalizar_quita_acentos_pero_no_la_enie():
    assert dt.normalizar("Inflación AÑO") == "inflacion año"


def test_dataset_de_ejemplo_pasa_la_regla_de_oro():
    total, tirados = dt.validar_jsonl(DATASET)
    assert total == 3 and tirados == []


def test_validar_ejemplo_sin_assistant():
    fila = {"messages": dt.mensajes(NOTAS)}
    assert dt.validar_ejemplo(fila) == ["faltan roles ['assistant']"]


def test_notas_desde_items_roundtrip():
    items = [{"title": "Banxico mantiene la tasa", "desc": "Sin cambio.", "link": "u"}]
    bloque = dt.notas_desde_items(items, "BBC Mundo", "2026-10-02")
    (nota,) = dt.parse_notas(bloque)
    assert nota.medio == "BBC Mundo" and nota.titulo == "Banxico mantiene la tasa"


def test_mensajes_forma_de_chat():
    fila = json.loads(DATASET.read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in fila["messages"]] == ["system", "user", "assistant"]
    assert fila["messages"][0]["content"] == dt.SYSTEM
