"""src/app/fallo_imagen_IA_pipeline.py — Un fallo del pipeline viaja como error, no como éxito.

Hasta el 17-sep-2026, cuando `TiendasProcess.process_frame` atrapaba una
excepción devolvía `{'error': mensaje}` como metadata, y `_full_frame_sync`
lo respondía igual que un frame bueno: `status: "success"`, sin `detections`
y, en modo directo, sin imagen. El cliente no tenía nada que pintar y no
repintaba, pero el contador de FPS de IA seguía subiendo porque las respuestas
sí llegaban. Resultado: imagen congelada con la consola del cliente limpia, y
el error visible solo en la del servidor. Así estuvo el detector caído por
«No module named 'vigilante_amazonas'» sin que quien miraba el cliente
pudiera saberlo.

El cliente ya sabía mostrar un `status: "error"` (imprime su `message` por
consola); lo que faltaba era que el servidor lo mandara.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def respuesta_si_fallo(metadata: Any, camera_id: Any,
                       processing_time: float) -> Optional[Dict[str, Any]]:
    """La respuesta de error si `metadata` es un fallo atrapado; si no, None.

    Se reconoce la forma EXACTA que devuelve el `except` de
    `TiendasProcess.process_frame`: un dict con la clave `error` y nada más.
    No basta con que exista la clave: la misma imagen levanta los servidores
    de otras verticales, y un pipeline que anotara un `error` informativo
    junto a resultados válidos no puede convertirse en un frame fallido —
    eso congelaría al cliente en vez de destapar el problema.
    """
    if not isinstance(metadata, dict) or set(metadata) != {"error"}:
        return None
    return {
        "camera_id": camera_id,
        "status": "error",
        # El cliente lo imprime tal cual (`data.get("message", ...)`).
        "message": str(metadata["error"] or "el pipeline fallo sin mensaje"),
        "metadata": metadata,
        "processed_image": None,
        "processing_time": processing_time,
    }
