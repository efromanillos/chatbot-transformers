# ============================================================
# historial.py
# Módulo funcional para gestionar el historial del chatbot
# con soporte para "contexto activo" por tema.
# ============================================================

# Historial completo de la conversación
_historial = []

# Contexto activo (solo turnos del tema actual)
_contexto_activo = []


def agregar_turno(pregunta: str, respuesta: str, dependiente: bool):
    """
    Añade un turno al historial completo y actualiza el contexto activo.
    Si la pregunta NO es dependiente, se inicia un nuevo contexto.
    """
    turno = {
        "usuario": pregunta,
        "bot": respuesta
    }

    # Guardar siempre en el historial completo
    _historial.append(turno)

    # Actualizar contexto activo
    if dependiente:
        # Continuamos el mismo tema
        _contexto_activo.append(turno)
    else:
        # Nuevo tema → reiniciar contexto activo
        _contexto_activo.clear()
        _contexto_activo.append(turno)


def obtener_historial():
    """Devuelve el historial completo."""
    return _historial


def limpiar_historial():
    """Limpia el historial completo y el contexto activo."""
    _historial.clear()
    _contexto_activo.clear()


def construir_consulta_con_contexto(pregunta: str) -> str:
    """
    Construye una consulta enriquecida usando SOLO el contexto activo.
    Se usa únicamente cuando la pregunta es dependiente.
    """
    if not _contexto_activo:
        return pregunta

    contexto = ""

    for turno in _contexto_activo:
        contexto += " " + turno["usuario"]

    consulta = contexto.strip() + " " + pregunta
    return consulta.strip()


def es_pregunta_dependiente(pregunta: str) -> bool:
    """
    Detecta si la pregunta depende del contexto anterior.
    SIN list comprehension.
    """
    dependencias = [
        "y ", "también", "además", "entonces",
        "él", "ella", "eso", "esa", "ese",
        "en qué año", "cuándo fue", "dónde fue",
        "qué hizo", "qué pasó", "y luego", "y después",
        "quién la", "quién lo", "quién fue", "en qué consiste"
    ]

    p = pregunta.lower()

    for d in dependencias:
        if d in p:
            return True

    return False