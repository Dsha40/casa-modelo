import sys, os, json, base64, time
import time
from datetime import datetime
import win32gui
import win32ui
import win32con
from PIL import Image
import ctypes
from ctypes import wintypes
import io

# 🔥 SUPRIMIR LOGS NO DESEADOS AL INICIO
import warnings
warnings.filterwarnings('ignore')

# Configurar variables de entorno para Qt ANTES de cualquier import
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.*=false'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

# Configurar PrintWindow
PrintWindow = ctypes.windll.user32.PrintWindow
PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
PrintWindow.restype = wintypes.BOOL

# Cap de ANCHO del frame que se envia al servidor. El process_frame del
# servidor escala ~cuadraticamente con los pixeles (deteccion + crops de
# demografia + draw): a 1920px ~210ms/frame (5 fps), a 1280px ~130ms,
# a 960px ~87ms (11 fps). Reducir el ancho dispara el FPS. En camaras
# lejanas (caras pequenas que ya no se detectan) NO se pierde demografia.
# Subir para mas nitidez/precision facial; bajar para mas FPS.
# Override por variable de entorno: CAPTURE_MAX_WIDTH.
# (Los tiempos se midieron con el detector YOLO de entonces; desde el
# 3-sep-2026 detecta RF-DETR y los absolutos cambian, pero la relacion
# "menos pixeles = mas fps" es la misma y es lo que decide este cap.)
try:
    MAX_SEND_WIDTH = int(os.environ.get("CAPTURE_MAX_WIDTH", "1600"))
except ValueError:
    MAX_SEND_WIDTH = 1600


def _entero_de_entorno(nombre: str, por_defecto: int) -> int:
    try:
        return int(os.environ.get(nombre, str(por_defecto)))
    except ValueError:
        return por_defecto


# El bucle duerme 1/60 s, asi que estos contadores son ~segundos x 60.
# Fallos SEGUIDOS antes de avisar de que la ventana no se puede capturar.
FALLOS_PARA_AVISAR = _entero_de_entorno("CAPTURE_FALLOS_AVISO", 180)      # ~3 s
# Fallos SEGUIDOS antes de volver a elegir el metodo de captura desde cero.
FALLOS_PARA_REPROBAR = _entero_de_entorno("CAPTURE_FALLOS_REPROBAR", 600)  # ~10 s
# Segundos con la MISMA imagen antes de avisar de que esta congelada. Una
# escena nocturna quieta puede repetirse, pero no bit a bit durante minutos:
# si eso pasa, o la aplicacion dejo de redibujar o se captura algo muerto.
SEG_PARA_AVISAR_CONGELADA = _entero_de_entorno("CAPTURE_SEG_CONGELADA", 120)
# Cada cuantos frames se comprueba si la imagen cambio (el hash cuesta).
CADA_CUANTOS_COMPROBAR = _entero_de_entorno("CAPTURE_CADA_CONGELADA", 30)


def _avisar(mensaje: str) -> None:
    """Escribe un aviso por stderr, que es lo unico que el padre puede leer.

    El worker NO puede escribir en stdout: ahi van los frames en msgpack y
    cualquier texto rompe el flujo. Y no puede quedarse callado: una captura
    muerta costo 19 h de deteccion el 8-ago-2026 sin una sola linea en ningun
    registro. Nunca lanza: avisar jamas puede tumbar la captura.
    """
    try:
        sys.stderr.write(f"[captura] {mensaje}\n")
        sys.stderr.flush()
    except Exception:
        pass




def _copiar_con_bitblt(saveDC, mfcDC, width, height) -> bool:
    """Copia la ventana con BitBlt. True si copio algo.

    ## Por que existe esta funcion de una linea

    `PyCDC.BitBlt` de win32ui **devuelve `None` cuando tiene EXITO** y avisa de
    los fallos LANZANDO `win32ui.error`. El codigo hacia:

        result = saveDC.BitBlt(...)      # None aunque copie perfectamente
        if result:                       # -> False SIEMPRE
            ...usar la imagen...
        else:
            return None                  # <- se tiraba una captura buena

    Es decir: el respaldo de BitBlt **nunca funciono**. Mientras PrintWindow
    respondia no se notaba; el dia que PrintWindow empezo a fallar (8-ago-2026,
    medido 0 exitos de 8 en las dos ventanas del iVMS-4200, mientras BitBlt
    daba 8 de 8 con imagen VIVA) la captura se quedo en None y el recuadro se
    congelo sin respaldo y sin aviso.
    """
    try:
        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
        return True
    except Exception:
        return False


def capture_window_by_hwnd(hwnd, pw_flag=2):
    try:
        # Obtener dimensiones de la ventana
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        # Obtener el contexto del dispositivo de la ventana
        hwndDC = win32gui.GetWindowDC(hwnd)
        if not hwndDC:
            return None

        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # Crear bitmap compatible
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        # Intentar capturar con PrintWindow (flag elegido por el caller)
        result = PrintWindow(hwnd, saveDC.GetSafeHdc(), pw_flag)

        if not result:
            result = _copiar_con_bitblt(saveDC, mfcDC, width, height)

        if result:
            # Convertir a formato PIL Image
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            im = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )



            # Limpiar recursos
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

            return im
        else:
            # Limpiar recursos en caso de error
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            return None

    except Exception:
        return None


def pil_image_to_png_bytes(imagen_pil, format="PNG", quality=None):
    try:
        buffer = io.BytesIO()
        is_quality = quality if quality else None
        imagen_pil.save(buffer, format=format, quality=is_quality)
        png_bytes = buffer.getvalue()
        buffer.close()
        return png_bytes
    except Exception:
        return None


def _looks_blank(im):
    """True si la imagen es casi uniforme (negra/vacia): senal de que
    PrintWindow con flag 0 no capturo el contenido de una app GPU."""
    try:
        lo, hi = im.convert('L').resize((32, 32)).getextrema()
        return (hi - lo) < 8
    except Exception:
        return False


import msgpack  # Agrega esta línea para serialización binaria


def _escribir_salida(data: bytes) -> None:
    """Escribe los bytes msgpack al pipe del proceso padre.

    OJO empaquetado: con PyInstaller en modo ventana (console=False)
    `sys.stdout` puede ser None, pero el descriptor 1 heredado del padre
    (QProcess crea el pipe) SÍ es válido. Por eso se cae a os.write(1).
    """
    try:
        salida = getattr(sys.stdout, "buffer", None)
        if salida is not None:
            salida.write(data)
            sys.stdout.flush()
            return
    except Exception:
        pass
    os.write(1, data)


def ejecutar_worker(hwnd: int) -> None:
    """Bucle de captura de la ventana `hwnd`: emite frames JPEG por stdout
    (msgpack) hasta que el proceso padre lo cierre.

    Se usa de DOS formas:
      - en desarrollo:  python src/workers/capture_woker.py <hwnd>
      - empaquetado:    PerimetralesView.exe --capture-worker <hwnd>
        (el exe se re-invoca a sí mismo; ver main.py. Sin esto, lanzar
         sys.executable abriría otra copia de la aplicación completa.)
    """
    try:
        # Flag de PrintWindow auto-seleccionado por ventana:
        #   0 = GDI estandar (~3x mas rapido: ~11ms vs ~33ms) para apps normales.
        #   2 = PW_RENDERFULLCONTENT, necesario para apps con GPU (Chrome/Brave/
        #       DirectX) que con flag 0 saldrian en negro.
        # Probamos flag 0 primero; si sale "en blanco" caemos a flag 2 y
        # cacheamos la decision (no se recalcula en cada frame).
        pw_flag = None
        # Fallos SEGUIDOS con el flag ya elegido. Un flag que funcionaba puede
        # dejar de hacerlo sin que la ventana cambie: PrintWindow le manda un
        # WM_PRINT a la aplicacion y ESPERA, asi que con la app ocupada empieza
        # a agotar el plazo. Antes se cacheaba la decision para siempre y la
        # captura no volvia sola nunca (8-ago-2026: 19 h congelada). Ahora, tras
        # un rato fallando, se vuelve a probar desde cero.
        fallos = 0
        avisado = False
        # Vigilancia de imagen congelada.
        frames_vistos = 0
        ultima_firma = None
        cambio_en = time.time()
        congelada = False

        while True:
            if pw_flag is None:
                buffer = capture_window_by_hwnd(hwnd, 0)
                if buffer is None or _looks_blank(buffer):
                    alt = capture_window_by_hwnd(hwnd, 2)
                    if alt is not None and not _looks_blank(alt):
                        pw_flag, buffer = 2, alt      # app GPU: fijar flag 2
                    elif buffer is None:
                        buffer = alt
                    # si ambos salen en blanco la ventana aun no renderiza:
                    # no fijamos pw_flag y reintentamos el proximo frame.
                else:
                    pw_flag = 0                       # app normal: flag 0 rapido
            else:
                buffer = capture_window_by_hwnd(hwnd, pw_flag)

            if buffer is None:
                fallos += 1
                if fallos == FALLOS_PARA_AVISAR and not avisado:
                    avisado = True
                    _avisar(f'sin captura de la ventana {hwnd} desde hace '
                            f'{FALLOS_PARA_AVISAR / 60:.0f} s')
                if fallos >= FALLOS_PARA_REPROBAR:
                    # Volver a elegir el metodo desde cero, por si la ventana
                    # cambio de forma de dibujar.
                    pw_flag, fallos = None, 0
            else:
                if avisado:
                    _avisar(f'captura de la ventana {hwnd} RECUPERADA')
                    avisado = False
                fallos = 0
                # ¿La imagen se quedo quieta? Capturar bien y capturar algo
                # VIVO no es lo mismo: se puede estar copiando durante horas
                # el ultimo fotograma que la aplicacion dejo dibujado.
                frames_vistos += 1
                if frames_vistos % CADA_CUANTOS_COMPROBAR == 0:
                    firma = hash(buffer.tobytes())
                    ahora = time.time()
                    if firma != ultima_firma:
                        ultima_firma, cambio_en = firma, ahora
                        if congelada:
                            _avisar(f'la ventana {hwnd} volvio a MOVERSE')
                            congelada = False
                    elif (not congelada
                          and ahora - cambio_en >= SEG_PARA_AVISAR_CONGELADA):
                        congelada = True
                        _avisar(f'la ventana {hwnd} lleva '
                                f'{int(ahora - cambio_en)} s dando la MISMA '
                                'imagen: se captura bien pero no cambia nada')

            # Cap de resolucion: reduce el ancho antes de codificar/enviar.
            # Acelera encode + red + process_frame del servidor.
            if buffer is not None and MAX_SEND_WIDTH > 0 and buffer.width > MAX_SEND_WIDTH:
                _nh = max(1, int(buffer.height * MAX_SEND_WIDTH / buffer.width))
                buffer = buffer.resize((MAX_SEND_WIDTH, _nh))

            if buffer:
                image_bytes = pil_image_to_png_bytes(buffer, 'JPEG', 70)
                if image_bytes:
                    header = {
                        "timestamp": datetime.now().isoformat(),
                        "size": len(image_bytes),
                        "format": "JPEG"
                    }

                    # Serializar mensaje binario con MessagePack
                    message = {
                        'header': header,
                        'image_bytes': image_bytes
                    }
                    binary_message = msgpack.packb(message)

                    # Enviar bytes directamente al pipe del padre
                    _escribir_salida(binary_message)

            time.sleep(1 / 60)

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        # 🔥 SILENCIAR CUALQUIER EXCEPCIÓN EN PRODUCCIÓN
        pass


# 🔥 MANEJO SEGURO DE ARGUMENTOS (modo script, en desarrollo)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    try:
        ejecutar_worker(int(sys.argv[1]))
    except (ValueError, TypeError):
        sys.exit(1)
