"""Router multimodal de percepcion para ELDE.

Enruta una consulta (imagen + texto) al motor adecuado, bajo el esquema:

  - "¿Que hay en esta imagen?"  -> VQA            -> Qwen2.5-VL-3B (nativo en transformers)
  - "Encuentra X / detecta Y"   -> Grounding/det. -> OWLv2 (open-vocab, Apache-2.0)
  - tracking + zonas + viz                         -> lo hace tiendas + Supervision

El grounding es OWLv2 (Apache-2.0) con pesos locales; el 3-sep-2026 se
retiro el rollback a YOLO-World y con el ultralytics (AGPL) desaparecio de
este modulo y de la imagen: ELDE_WORLD_MOTOR admite solo 'owlv2' | 'off'
(ver _ensure_world). Los modelos se cargan de forma PEREZOSA la primera vez
que se usan (asi el arranque del servidor de tiempo real no espera por
ellos). Caben residentes en 16GB: OWLv2-base (~0.4GB fp16) + Qwen2.5-VL-3B
(~7GB).
"""

import logging
import os
import re
import threading
import time

logger = logging.getLogger(__name__)

# Palabras que indican intencion de DETECCION/grounding (vs pregunta abierta).
_DETECT_KW = (
    "detecta", "detectar", "deteccion", "encuentra", "encontrar", "localiza",
    "localizar", "ubica", "ubicar", "busca", "buscar", "marca", "señala",
    "senala", "cuenta", "cuantos", "cuántos", "cuantas", "cuántas",
    "find", "detect", "locate", "show me all", "where is", "where are",
    "todas las", "todos los",
)
# Articulos/inicios a recortar al extraer el termino a detectar.
_LEAD_STRIP = re.compile(
    r"^\s*(el|la|los|las|un|una|unos|unas|all|the|a|an)\s+", re.IGNORECASE)


class MotorMundoOWLv2:
    """Grounding open-vocab con OWLv2 (owlv2-base-patch16-ensemble, Apache-2.0).

    El UNICO motor open-vocab desde el 3-sep-2026 (sustituyo al AGPL, ya
    retirado, conservando su contrato de salida
    [{box:[x1,y1,x2,y2] px, label, confidence}]). Pesos LOCALES bajo
    server/models/world (montada :ro en los contenedores) y
    local_files_only=True SIEMPRE: un servidor de produccion no sale a
    internet a mitad de una busqueda. La envoltura es ligera; los pesos se
    cargan perezosos CON candado (los endpoints /vlm corren en threadpool)
    y en fp16 sobre CUDA.

    OJO calibracion: los scores de OWLv2 corren MAS BAJOS que los del motor
    anterior. El umbral que llega se respeta tal cual, pero un knob afinado
    para aquel (p.ej. ELDE_VLM_BUSQUEDA_CONF=0.25) filtra de mas: con OWLv2
    el valor sano ronda ~0.12.
    """

    #: `server/src/analityc/core/` -> tres niveles arriba es `server/`.
    RUTA_PESOS = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "models", "world", "owlv2-base-patch16-ensemble"))

    def __init__(self, device=0):
        self.device = device
        self._lock = threading.Lock()
        self._proc = None
        self._modelo = None

    def _load(self):
        """Carga perezosa con doble chequeo (varios hilos piden a la vez)."""
        if self._modelo is not None:
            return
        with self._lock:
            if self._modelo is not None:
                return
            # config.json Y el fichero de pesos, no solo la carpeta: una
            # descarga a medias no debe pasar por "pesos presentes".
            completo = (os.path.isfile(
                os.path.join(self.RUTA_PESOS, "config.json"))
                and any(os.path.isfile(os.path.join(self.RUTA_PESOS, f))
                        for f in ("model.safetensors", "pytorch_model.bin")))
            if not completo:
                raise RuntimeError(
                    f"pesos OWLv2 ausentes o incompletos: {self.RUTA_PESOS} "
                    "— descarga google/owlv2-base-patch16-ensemble ahi, o "
                    "apaga la busqueda open-vocab con ELDE_WORLD_MOTOR=off")
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            t0 = time.time()
            cpu = str(self.device) == "cpu"
            dev = "cpu" if cpu else f"cuda:{self.device}"
            proc = Owlv2Processor.from_pretrained(
                self.RUTA_PESOS, local_files_only=True)
            modelo = Owlv2ForObjectDetection.from_pretrained(
                self.RUTA_PESOS, local_files_only=True,
                dtype=torch.float32 if cpu else torch.float16).to(dev).eval()
            # _proc primero: el chequeo rapido de arriba mira _modelo, que
            # solo se publica cuando todo esta listo.
            self._proc = proc
            self._modelo = modelo
            logger.info("OWLv2 cargado (%s en %s) en %.1fs",
                        os.path.basename(self.RUTA_PESOS), dev,
                        time.time() - t0)

    def detect(self, image_bgr, terms, conf=0.05):
        """BGR + terminos -> [{box, label, confidence}] en pixeles."""
        self._load()
        import cv2
        import torch
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        entra = self._proc(text=[terms], images=img, return_tensors="pt")
        entra = {k: v.to(self._modelo.device) for k, v in entra.items()}
        # El modelo puede estar en fp16: los pixeles viajan en SU dtype.
        entra["pixel_values"] = entra["pixel_values"].to(self._modelo.dtype)
        with torch.no_grad():
            sale = self._modelo(**entra)
        alto, ancho = image_bgr.shape[:2]
        res = self._proc.post_process_grounded_object_detection(
            sale, threshold=float(conf), target_sizes=[(alto, ancho)])[0]
        out = []
        for caja, score, idx in zip(res["boxes"], res["scores"],
                                    res["labels"]):
            # El preproceso de OWLv2 rellena la imagen a cuadrado (abajo/
            # derecha) y el reescalado usa el lado mayor: una caja puede
            # asomarse al relleno -> se recorta al marco real.
            x1, y1, x2, y2 = (float(v) for v in caja.tolist())
            i = int(idx)
            out.append({
                "box": [min(max(x1, 0.0), float(ancho)),
                        min(max(y1, 0.0), float(alto)),
                        min(max(x2, 0.0), float(ancho)),
                        min(max(y2, 0.0), float(alto))],
                "label": terms[i] if i < len(terms) else str(i),
                "confidence": float(score),
            })
        return out


class MultimodalRouter:
    """Carga perezosa del motor open-vocab (OWLv2, Apache-2.0) + Qwen2.5-VL
    (VQA) y enrutado."""

    # System prompt que enmarca la tarea como analitica autorizada -> evita que
    # Qwen2.5-VL rechace describir/contar personas (guardrail de privacidad).
    VQA_SYSTEM = (
        "Eres un asistente de analitica de video retail/seguridad autorizado. "
        "Describe objetivamente la escena (lugar, numero de personas y su "
        "actividad) con fines de negocio. Responde en espanol, sin rechazar.")

    # Modelos VQA seleccionables. Se mantiene UNO cargado a la vez: al cambiar se
    # libera el actual y el nuevo se carga en la proxima consulta (ahorra VRAM,
    # solo se carga el que se necesita). key -> (model_id, int8).
    VQA_MODELS = {
        "3b": ("Qwen/Qwen2.5-VL-3B-Instruct", False),  # bf16 (~7GB) rapido
        "7b": ("Qwen/Qwen2.5-VL-7B-Instruct", True),   # int8 (~8.8GB) calidad
    }
    # Persistencia del modelo activo y del contexto (sobreviven reinicios).
    _STATE_FILE = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "output", "vlm_model.txt"))
    _CONTEXT_FILE = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "output", "vlm_context.txt"))

    #: Valores admitidos de ELDE_WORLD_MOTOR. El 'yoloworld' de rollback se
    #: retiro el 3-sep-2026 junto con ultralytics (AGPL); si alguien lo deja
    #: en un .env viejo, _ensure_world avisa y usa owlv2 (no revienta).
    MOTORES_WORLD = ("owlv2", "off")

    def __init__(self, device=0, vqa_model=None):
        self.device = device
        self._world = None             # MotorMundoOWLv2 (None si esta 'off')
        self._world_lock = threading.Lock()
        self._world_off_avisado = False
        self._world_motor_avisado = False   # valor no reconocido: 1 aviso
        self._vqa_models = {}          # key -> (model, proc); solo uno a la vez
        self._vqa_lock = threading.Lock()
        self._vqa_key = self._load_persisted_key(vqa_model)
        self.context = self._load_context()   # contexto GLOBAL del negocio/escena

    # ── persistencia del modelo activo ───────────────────────────
    def _load_persisted_key(self, override):
        if override in self.VQA_MODELS:
            return override
        try:
            with open(self._STATE_FILE, "r", encoding="utf-8") as f:
                k = f.read().strip().lower()
                if k in self.VQA_MODELS:
                    return k
        except Exception:
            pass
        return "7b"

    def _persist_key(self, key):
        try:
            os.makedirs(os.path.dirname(self._STATE_FILE), exist_ok=True)
            with open(self._STATE_FILE, "w", encoding="utf-8") as f:
                f.write(key)
        except Exception:
            pass

    # ── contexto del negocio/escena ──────────────────────────────
    def _load_context(self):
        # 1) archivo persistido (set en runtime); 2) config; 3) vacio.
        try:
            with open(self._CONTEXT_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
        try:
            from .analytics.config import AnalyticsConfig
            return (getattr(AnalyticsConfig, "VLM_CONTEXT", "") or "").strip()
        except Exception:
            return ""

    def set_context(self, text):
        """Fija el contexto GLOBAL (se inyecta en todas las consultas). Persiste."""
        self.context = (text or "").strip()
        try:
            os.makedirs(os.path.dirname(self._CONTEXT_FILE), exist_ok=True)
            with open(self._CONTEXT_FILE, "w", encoding="utf-8") as f:
                f.write(self.context)
        except Exception:
            pass
        return True

    def _context_block(self, extra=None):
        """Bloque de contexto que se añade al system prompt: contexto GLOBAL +
        el 'extra' que venga POR consulta. Vacio si no hay nada."""
        parts = [p.strip() for p in (self.context, extra)
                 if p and str(p).strip()]
        if not parts:
            return ""
        return "\n\nContexto de la escena:\n" + "\n".join(parts)

    # ── carga ────────────────────────────────────────────────────
    def _ensure_world(self):
        """Motor open-vocab segun ELDE_WORLD_MOTOR (se lee en cada llamada):

          - 'owlv2' (default): OWLv2 con pesos locales (Apache-2.0). La
            envoltura es ligera; los pesos cargan en el primer detect() o
            en warmup().
          - 'off': devuelve None -> detect() responde [] con un aviso una
            sola vez (busqueda_vlm ya degrada con cero resultados).
          - cualquier otro valor (el 'yoloworld' de rollback se retiro el
            3-sep-2026 con ultralytics): avisa UNA vez y usa owlv2. Un .env
            viejo no puede dejar la busqueda sin motor.
        """
        motor = ((os.getenv("ELDE_WORLD_MOTOR") or "").strip().lower()
                 or "owlv2")
        if motor == "off":
            if not self._world_off_avisado:
                logger.warning("ELDE_WORLD_MOTOR=off: deteccion open-vocab "
                               "apagada (detect devuelve [])")
                self._world_off_avisado = True
            return None
        if motor not in self.MOTORES_WORLD and not self._world_motor_avisado:
            logger.warning(
                "ELDE_WORLD_MOTOR=%r no reconocido (validos: %s; 'yoloworld' "
                "se retiro el 3-sep-2026 con ultralytics): se usa owlv2",
                motor, " | ".join(self.MOTORES_WORLD))
            self._world_motor_avisado = True
        if self._world is None:
            with self._world_lock:
                if self._world is None:
                    self._world = MotorMundoOWLv2(device=self.device)
        return self._world

    def _load_one_vqa(self, key):
        """Carga (si falta) y devuelve (model, proc) del modelo 'key'."""
        if key in self._vqa_models:
            return self._vqa_models[key]
        import torch
        from transformers import (Qwen2_5_VLForConditionalGeneration,
                                  AutoProcessor)
        mid, int8 = self.VQA_MODELS[key]
        t0 = time.time()
        cpu = str(self.device) == "cpu"
        dev = "cpu" if cpu else f"cuda:{self.device}"
        if int8 and not cpu:
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_8bit=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mid, quantization_config=qc, dtype=torch.bfloat16,
                device_map=dev).eval()
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mid, dtype=torch.bfloat16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(mid)
        self._vqa_models[key] = (model, proc)
        logger.info("VQA cargado: %s (%s, int8=%s) en %.0fs",
                    key, mid, int8, time.time() - t0)
        return self._vqa_models[key]

    def _free_all_vqa_locked(self):
        """Libera el/los modelo(s) VQA cargado(s) (con self._vqa_lock tomado)."""
        if self._vqa_models:
            self._vqa_models.clear()
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def set_vqa_model(self, key):
        """Selecciona el modelo VQA ('3b' | '7b'). Solo se mantiene UNO cargado:
        al cambiar libera el actual y el nuevo se carga al usarse. Persiste."""
        key = str(key).lower().strip()
        if key not in self.VQA_MODELS:
            return False
        with self._vqa_lock:
            if key != self._vqa_key:
                self._free_all_vqa_locked()
            self._vqa_key = key
        self._persist_key(key)
        logger.info("VQA seleccionado -> %s (se carga al usarse)", key)
        return True

    def current_vqa(self):
        return {
            "key": self._vqa_key,
            "model_id": self.VQA_MODELS[self._vqa_key][0],
            "loaded": list(self._vqa_models.keys()),
            "available": list(self.VQA_MODELS.keys()),
        }

    def warmup(self, what=("world", "vqa")):
        """Pre-carga opcional. 'vqa' carga SOLO el modelo activo."""
        if "world" in what:
            m = self._ensure_world()
            if m is not None:           # None solo con ELDE_WORLD_MOTOR=off
                m._load()   # la envoltura es ligera: los pesos cargan aqui
        if "vqa" in what:
            with self._vqa_lock:
                self._load_one_vqa(self._vqa_key)

    # ── motores ──────────────────────────────────────────────────
    def detect(self, image_bgr, terms, conf=0.05):
        """Open-vocab detection con OWLv2 (Apache-2.0; desde el 3-sep-2026
        el unico motor: el rollback AGPL se retiro). terms: str o lista de
        clases. Devuelve [{box:[x1,y1,x2,y2], label, confidence}] en pixeles.
        """
        m = self._ensure_world()
        if isinstance(terms, str):
            terms = [terms]
        terms = [t for t in (s.strip() for s in terms) if t]
        if not terms or m is None:   # sin terminos, o ELDE_WORLD_MOTOR=off
            return []
        return m.detect(image_bgr, terms, conf=conf)

    def vqa(self, image_bgr, question, max_new_tokens=160, system=None,
            context=None):
        """Pregunta abierta sobre la imagen con Qwen2.5-VL. Devuelve texto.
        Inyecta el contexto GLOBAL + el 'context' por consulta en el system
        prompt (analitica autorizada; evita el guardrail que rechaza personas)."""
        import cv2
        import torch
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        sys_msg = (system or self.VQA_SYSTEM) + self._context_block(context)
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question}]}]
        # Lock: serializa inferencia (un generate a la vez sobre la GPU).
        with self._vqa_lock:
            model, proc = self._load_one_vqa(self._vqa_key)
            text = proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[img],
                       return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=max_new_tokens,
                                   do_sample=False)
            ans = proc.batch_decode(
                [g[0][inp["input_ids"].shape[1]:]], skip_special_tokens=True)[0]
        return ans.strip()

    # ── deteccion de eventos (compra / entrega de bandeja) ───────
    EVENTS_PROMPT = (
        "Analiza la imagen de la camara de un local de comida. Determina si "
        "esta ocurriendo alguno de estos eventos:\n"
        "1. COMPRA: una persona realizando una compra o pago (en caja o "
        "mostrador, entregando dinero o tarjeta, siendo atendida por un "
        "cajero).\n"
        "2. ENTREGA_BANDEJA: una persona entregando o recibiendo una bandeja "
        "con comida.\n"
        "Responde UNICAMENTE con un JSON valido, sin texto adicional, con esta "
        "forma exacta:\n"
        '{"compra": true|false, "entrega_bandeja": true|false, '
        '"confianza": "alta|media|baja", "descripcion": "<breve, 1 frase>"}')

    def detect_events(self, image_bgr, max_new_tokens=200, context=None):
        """Detecta 'compra' y 'entrega de bandeja de comida' con el VLM.
        Devuelve dict con flags booleanos + descripcion. 'context' por consulta
        se suma al contexto global."""
        import json
        import re
        raw = self.vqa(image_bgr, self.EVENTS_PROMPT,
                       max_new_tokens=max_new_tokens, context=context)
        data = {}
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                # tolera comillas raras / true-false en mayuscula
                txt = m.group(0).replace("True", "true").replace("False", "false")
                try:
                    data = json.loads(txt)
                except Exception:
                    data = {}
        return {
            "compra": bool(data.get("compra", False)),
            "entrega_bandeja": bool(data.get("entrega_bandeja", False)),
            "confianza": data.get("confianza", "baja"),
            "descripcion": str(data.get("descripcion", "")).strip(),
            "raw": raw,
        }

    # ── enrutado por intencion ───────────────────────────────────
    @staticmethod
    def looks_like_detection(query: str) -> bool:
        ql = (query or "").lower()
        return any(k in ql for k in _DETECT_KW)

    @staticmethod
    def extract_terms(query: str):
        """De 'encuentra el boton azul' -> ['boton azul'] (clase open-vocab)."""
        ql = (query or "").strip().rstrip("?.!¿¡")
        low = ql.lower()
        for k in _DETECT_KW:
            idx = low.find(k)
            if idx != -1:
                ql = ql[idx + len(k):].strip()
                break
        ql = _LEAD_STRIP.sub("", ql)
        ql = ql.strip()
        return [ql] if ql else ["object"]

    def route(self, image_bgr, query: str, conf=0.05, max_new_tokens=128,
              context=None):
        """Decide VQA vs grounding por la consulta y ejecuta."""
        if self.looks_like_detection(query):
            terms = self.extract_terms(query)
            return {"mode": "detect", "terms": terms,
                    "detections": self.detect(image_bgr, terms, conf=conf)}
        return {"mode": "vqa",
                "answer": self.vqa(image_bgr, query, max_new_tokens,
                                   context=context)}


class EventMonitor:
    """Deteccion CONTINUA de eventos (compra / entrega de bandeja) disparada por
    el detector del pipeline (RF-DETR). Cada frame se llama maybe_check(); si
    hay trigger (el detector vio persona) y paso el throttle, lanza
    detect_events() en un hilo de fondo (la
    inferencia VLM tarda ~20-30s y NO debe bloquear el bucle de tiempo real).
    Un solo VLM corre a la vez en TODO el servidor (lock global) para no chocar
    entre camaras ni reventar la VRAM. El resultado positivo se lee con
    consume() y se inyecta en metadata['alerts'] (el cliente ya lo muestra)."""

    _GLOBAL_LOCK = threading.Lock()

    def __init__(self, get_router, every_s=12.0):
        self._get_router = get_router
        self._every = float(every_s)
        self._last_check = 0.0
        self._pending = None
        self._lock = threading.Lock()

    def maybe_check(self, frame_bgr, has_trigger):
        now = time.time()
        if (not has_trigger or frame_bgr is None
                or (now - self._last_check) < self._every):
            return
        self._last_check = now
        if not EventMonitor._GLOBAL_LOCK.acquire(blocking=False):
            return  # ya hay un VLM event-check en curso (otra camara)
        try:
            f = frame_bgr.copy()
        except Exception:
            EventMonitor._GLOBAL_LOCK.release()
            return
        threading.Thread(target=self._run, args=(f,), daemon=True).start()

    def _run(self, frame):
        try:
            r = self._get_router().detect_events(frame)
            if r.get("compra") or r.get("entrega_bandeja"):
                import base64
                import cv2
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                r["image_base64"] = (base64.b64encode(buf).decode()
                                     if ok else "")
                with self._lock:
                    self._pending = r
        except Exception:
            logger.debug("EventMonitor error", exc_info=True)
        finally:
            EventMonitor._GLOBAL_LOCK.release()

    def consume(self):
        """Devuelve el ultimo evento positivo nuevo (una vez) o None."""
        with self._lock:
            r = self._pending
            self._pending = None
        return r


# ── Singleton compartido del router (carga perezosa) ─────────────────────────
# Lo usan los endpoints (app.py) y la deteccion de eventos (Hummus). Los modelos
# pesados se cargan la PRIMERA vez que se consulta (no retrasa el tiempo real).
_MM_ROUTER = None


def get_multimodal_router(device=0):
    """Devuelve el MultimodalRouter compartido (carga perezosa de OWLv2 +
    Qwen2.5-VL la 1a vez que se usa)."""
    global _MM_ROUTER
    if _MM_ROUTER is None:
        _MM_ROUTER = MultimodalRouter(device=device)
    return _MM_ROUTER
