@echo off
REM ============================================================
REM  INICIAR_SERVIDOR - Arranca SOLO el servidor de IA (ELDE):
REM    - websocket de inferencia   ws://0.0.0.0:9000/ws
REM    - dashboards                http://localhost:9000/dashboards/
REM    - panel VIGILANTE           http://localhost:5333
REM  Usa SIEMPRE el venv del servidor (con el Python global se levanta
REM  un servidor a medias que ocupa el puerto: gotcha del 4-ago-2026).
REM  No requiere Administrador. Cerrar: cerrar esta ventana o Ctrl+C.
REM ============================================================

setlocal
cd /d "%~dp0server"
title SERVIDOR ELDE (9000)

REM UTF-8: los print con acentos/emoji no deben crashear la consola.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Carpeta donde el servidor escribe la copia ANOTADA de cada captura para
REM la galeria del cliente Amazonas (pestana "Capturas" y panel lateral leen
REM esa carpeta local). El refactor a monorepo cambio la ruta escrita a mano
REM por esta variable y quedo SIN DEFINIR: el servidor guardaba las fotos en
REM server\output\captures pero el cliente no veia ninguna (7-ago-2026).
REM Solo vale con cliente y servidor en la misma maquina, que es el caso.
set "ELDE_CAPTURE_CLIENT_DIR=%~dp0clients\amazonas\capture"

REM ============================================================
REM  Ajustes de ESTA maquina (PC de pruebas, 72.68.60.171)
REM ============================================================

REM Panel del VIGILANTE en el 5333. La cabecera de arriba lo prometia desde
REM siempre, pero nadie encendia el interruptor: iniciar_servidor_headless.py
REM solo lo levanta si esta variable vale 1, y viene apagada por defecto
REM porque en el servidor Linux hay tres contenedores en red del host y
REM pelearian por el puerto. Aqui solo hay un proceso, asi que se enciende.
set "ELDE_PANEL_VIGILANTE=1"

REM Nombre con el que este servidor se anuncia en la baliza UDP 9010. Sin
REM esto usa el nombre de la maquina (DESKTOP-4VMGA5I) y en la ventana de
REM conexion de los clientes sale junto al de PRODUCCION (pop-os,
REM 72.68.60.141) sin nada que los distinga salvo la GPU. Los clientes NO
REM autoeligen de la baliza (arranque_servidor.py manda con --servidor, el
REM selector, la ultima eleccion o el .env), asi que esto no roba clientes:
REM solo evita que alguien pinche el equivocado. La identidad interna sigue
REM siendo el hostname, asi que baliza y /health responden lo mismo y ningun
REM cliente lo lista dos veces.
set "ELDE_SERVIDOR_NOMBRE=PRUEBAS-WINDOWS"

REM WhatsApp APAGADO a proposito. vigilante_amazonas/config.py trae
REM incrustada por defecto la URL del grupo REAL, y el envio se dispara solo
REM con que un cliente conecte con su interruptor "Enviar por WhatsApp"
REM puesto: probar perimetrales-view desde aqui mandaria fotos de prueba al
REM grupo de verdad. Se apunta a un puerto local cerrado; el emisor nunca
REM lanza, asi que el envio fallido no toca la inferencia. Para probar el
REM envio de VERDAD hay que borrar esta linea a sabiendas de a donde va.
set "WHATSAPP_BOT_URL=http://127.0.0.1:1/whatsapp-desactivado-en-pruebas"

REM Si ya hay alguien escuchando en 9000, no levantar un segundo servidor.
REM /c: NO es opcional: sin el, findstr parte el patron por el ESPACIO y
REM busca ":9000" O ".*LISTENING" por separado, asi que casa CUALQUIER
REM linea con LISTENING (las 60 de esta maquina) y tambien las
REM conexiones salientes al 9000 de otro servidor (un navegador abierto en
REM los paneles del de produccion basta). Resultado: el guardia saltaba
REM siempre y este .bat no podia arrancar el servidor nunca. Con /c: el
REM espacio forma parte del patron y solo casa la linea de escucha real.
netstat -ano | findstr /r /c:":9000 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo Ya hay un servidor escuchando en el puerto 9000.
    echo Si los clientes lo ven "offline", puede ser un servidor a medias
    echo lanzado con el Python global: cierralo y vuelve a ejecutar esto.
    echo.
    pause
    exit /b 1
)

echo Iniciando servidor de IA (la primera vez tarda ~1 min por los modelos)...
venv\Scripts\python.exe iniciar_servidor_headless.py 9000
pause
