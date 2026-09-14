"""
Red del ecosistema ELDE: encontrar el servidor de inferencia y recordarlo.

Tres piezas, una por fichero:

  descubrimiento_servidores.py  QUIEN hay en la red (baliza UDP + escaneo TCP)
  servidores_guardados.py       QUE recordamos entre arranques (manual + ultimo)
  arranque_servidor.py          CUAL usa este cliente al arrancar (la decision)

Todo el modulo es de STDLIB salvo un `psutil` opcional (solo para leer las
mascaras de red exactas). Es deliberado: estos ficheros viajan dentro de los
.exe distribuibles, donde cada dependencia cuesta megabytes y riesgo de que
PyInstaller no la empaquete.
"""
