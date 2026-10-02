import rss_reportero as rss

XML = b"""<?xml version="1.0"?>
<rss><channel>
  <item><title>Lluvias en CDMX</title><description><![CDATA[<p>Encharcamientos &amp; cierres</p>]]></description><link>a</link></item>
  <item><title>Deportes: gana el Tri</title><description>Amistoso</description><link>b</link></item>
  <item><title>Banxico y la economia</title><description>Tasa sin cambio</description><link>c</link></item>
  <item><title></title><description>sin titulo, se ignora</description></item>
</channel></rss>"""


def test_parse_items_limpia_html_y_salta_vacios():
    items = rss.parse_items(XML)
    assert len(items) == 3
    assert items[0]["desc"] == "Encharcamientos & cierres"


def test_filtro_antes_de_recortar():
    # Con --n 1, recortar primero dejaría "Lluvias" y el filtro devolvería vacío.
    bloque = rss.format_notes(rss.parse_items(XML), "economia", 1)
    assert "1. Banxico y la economia" in bloque


def test_feed_sin_coincidencias_avisa():
    bloque = rss.format_notes(rss.parse_items(XML), "terremoto", 5)
    assert "(el feed no trajo notas para ese filtro)" in bloque
