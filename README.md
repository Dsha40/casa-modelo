# ELDE — Tiendas

Analítica de visitantes para tienda: detección de personas, género y rango de
edad, re-identificación para no contar dos veces a la misma persona, mapa de
calor, anaqueles, permanencia y stand.

Son dos programas que hablan por websocket:

| | qué es | dónde |
|---|---|---|
| **Servidor** | FastAPI + PyTorch. Recibe fotogramas, los procesa en la GPU y devuelve las detecciones y la metadata | `server/` |
| **Cliente** | Aplicación de escritorio (PySide6). Se conecta al DVR, dibuja las zonas y manda los fotogramas | `clients/tienda/` |
| **Núcleo** | Lo que comparten los dos: contratos, DVR, descubrimiento, registro | `packages/elde_core/` |
| **Dashboards** | Las pantallas web que sirve el propio servidor | `dashboards/` |

## Qué NO está aquí

Este repositorio es **el corte de tiendas** de un sistema más grande. Fuera
quedan otros verticales (vigilancia perimetral, estacionamiento, comedor) y los
otros tres clientes de escritorio.

Eso se nota en dos sitios, y es normal, no es que esté roto:

- Al arrancar, el servidor avisa `vigilante_amazonas no disponible` y
  `perimetral_nocturna no disponible`. Sigue funcionando: esos modos
  simplemente no los sirve.
- Quince pruebas salen como `skipped` por la misma razón (mira
  `server/tests/conftest.py`).

**No hace falta rellenar esos huecos.** Si algo que necesitas para tiendas
parece depender de una pieza ausente, dilo antes de reconstruirla.

## Montarlo (Windows, una vez)

Hace falta **Python 3.12** y una **GPU NVIDIA** con su driver. Son unos 6 GB de
descarga: torch para CUDA 12.8 pesa lo suyo.

```powershell
python -m venv server\venv
server\venv\Scripts\python.exe -m pip install -U pip wheel
server\venv\Scripts\python.exe -m pip install -r deploy\requirements_servidor_ubuntu.txt
server\venv\Scripts\python.exe -m pip install --no-deps -e packages\elde_core
```

Y un paso que **no es opcional**: `easyocr` arrastra `opencv-python-headless`,
que pisa el `cv2` de contrib y deja fuera la mitad de los módulos. Hay que
quitarlo:

```powershell
server\venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless
server\venv\Scripts\python.exe -m pip install --no-deps --force-reinstall opencv-contrib-python==4.12.0.88
```

Comprobar que quedó bien (esta línea es la misma con la que se valida la imagen
de producción; si pasa, el entorno sirve):

```powershell
server\venv\Scripts\python.exe -c "import cv2, torch, onnxruntime, rfdetr, transformers, supervision, torchreid; assert hasattr(cv2,'ximgproc'), 'cv2 sin contrib'; print('cv2', cv2.__version__, '| torch', torch.__version__)"
server\venv\Scripts\python.exe server\scripts\verificar_entorno.py   # debe salir con código 0
```

### Los pesos

Van por **Git LFS** (unos 1,2 GB en `server/models`). Instálalo **antes** de
clonar: `git lfs install`. Instalar el binario no basta — si no registras los
filtros, el clon trae ficheros de 130 bytes con el nombre correcto y todo
parece bien hasta que no detecta nada.

Para comprobarlo: `server/models/base/rf-detr-medium.pth` debe pesar ~387 MB,
no 4 KB.

## Arrancarlo

```
INICIAR_SERVIDOR.bat     el servidor  (websocket :9000, dashboards y panel)
INICIAR_TIENDA.bat       el cliente
```

Con el servidor arriba: dashboards en `http://localhost:9000/dashboards/`,
salud en `/health`, API de lectura en `/api/v1`.

El cliente **no lleva la IP del servidor escrita**: lo busca en la red (baliza
UDP 9010) y, si no lo encuentra, lo pregunta.

## Las pruebas

```powershell
cd server
..\server\venv\Scripts\python.exe -m pytest tests -q
```

733 pasan hoy, 15 se saltan a propósito. **Que sigan pasando es la condición
para dar un cambio por bueno**: son el contrato de lo que ya funciona, y varias
fijan comportamientos que costaron caro de arreglar (mira las fechas en los
comentarios).

## Por dónde entrar al código

| Quieres tocar | Empieza por |
|---|---|
| El pipeline de tienda | `server/src/analityc/core/tiendas.py` (clase `TiendasProcess`) |
| Género y edad | `server/src/analityc/core/analytics/demographics.py` |
| Contar personas una sola vez | `server/src/analityc/core/analytics/face_reidentifier.py` |
| Anaqueles | `server/src/analityc/core/anaqueles.py` |
| Permanencia | `server/src/analityc/core/permanencia_area.py` |
| Stand | `server/src/analityc/core/amazonas_stand.py` |
| La ventana del cliente | `clients/tienda/src/gui/` |

**`tiendas.py` es la clase base** de anaqueles, permanencia y stand. Un cambio
ahí los toca a los tres: corre sus pruebas, no solo las de tienda.

Documentación de fondo: `README_STAND.md`, `README_PERMANENCIA.md`,
`CONTRATO_STAND.md`, `server/README_DEMOGRAFIA.md`.

## Cómo trabajar

Rama por tarea, y los cambios se entregan por **Pull Request**:

```bash
git checkout -b arreglo/lo-que-sea
# ...
git commit
git push -u origin arreglo/lo-que-sea
```

Dos costumbres del proyecto que conviene respetar:

1. **Los comentarios explican el porqué, no el qué.** Mira cualquier fichero:
   dicen qué se rompió, cuándo y cómo se midió. Eso es a propósito.
2. **Una funcionalidad, un fichero.** Lo nuevo va en su propio módulo, con el
   cableado mínimo en los que ya existen.
