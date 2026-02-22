#==================================================================================
# chatbot.py: 
# En este módulo se realiza:
# 1. Búsqueda semántica: similaridad del coseno
# 2. Generar la respuesta: se selcciona el chunk más relevante a la pregunta del usr
# 3. Responder: función orquestadora que encapsula las dos anteiores
#===================================================================================


import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#=================================================================
# buscar_similares()
# buscar imilitud entre pregunta y chunks, devulve lista de tuplas
# con los 3 chunks más similares
#=================================================================

def buscar_similares(pregunta: str, chunks: list[str], embeddings: np.ndarray, modelo: SentenceTransformer, k: int = 3) -> list[tuple[str,float]]:
    """Devuelve los k chunks más similares a la pregunt en una lista de tuplas (chunk, similitudes)."""
    
    # Embedding de la pregunta
    emb_pregunta = modelo.encode([pregunta])

    # Similaridad coseno
    similitudes = cosine_similarity(emb_pregunta, embeddings)[0]

    # Ordena los índices según la similitud (de menor a mayor),
    # luego los invierte [::-1] para obtenerlos de mayor a menor
    # y finalmente se queda con los k primeros [:k], es decir,
    # los 3 índices con mayor similitud.

    indices = np.argsort(similitudes)[::-1][:k]

    # Devolver lista de (chunk, similitud)
    resultado = []
    for i in indices:
        chunk = chunks[i]
        similitud = similitudes[i]
        resultado.append((chunk, similitud))
    return resultado


#===========================================
# generar_respuesta()
# Devuelve el chunk (str) con mayor similitud
#===========================================


def generar_respuesta(pregunta: str, contexto: list[str]) -> str:
    """
    Genera una respuesta.
    Devuelve el chunk más relevante como respuesta.
    """
    if not contexto:
        return 'No he encontrado información relevante en el documento.'

    # Tomamos el chunk más relevante
    mejor_chunk = contexto[0]

    # Respuesta simple
    respuesta = (
        'He encontrado la siguiente información relevante en el documento:\n\n'
        f'{mejor_chunk}\n\n'
        'Esta información se ha seleccionado por similitud semántica con tu pregunta.'
    )

    return respuesta

#=====================================================
# responder()
# Función orquestadora:
# Busca los 3 chunks más similares
# Devuelve un único chunk, el más relevante.
#=====================================================


def responder(pregunta: str, chunks: list[str], embeddings: np.ndarray, modelo: SentenceTransformer) -> str:
    """Orquesta todo el proceso"""
    
    similares = buscar_similares(pregunta, chunks, embeddings, modelo)
    contexto = []
    for chunk, _ in similares:
        #contexto contiene los 3 chunks más similares
        contexto.append(chunk)

    respuesta = generar_respuesta(pregunta, contexto)
    return respuesta