#====================================================================
#emebddings.py
#Módulo donde: 
# 1. Carga el modelo de embeddings
# 2. Divide en chunks el texto
# 3. Genera la matriz de embeddings
# 4. Encapsula todo en una función orquestadora preparar_emebddings()
#====================================================================


from sentence_transformers import SentenceTransformer
import numpy as np

#===============================
# 1. Cargar modelo de embeddings
#===============================

def cargar_modelo() -> SentenceTransformer:
    """Cargar el modelo de embeddings solo una vez"""
    modelo = SentenceTransformer('all-MiniLM-L6-v2')
    return modelo

#===============================================================
# 2. Dividir texto en chunks
# se especifica la salida tipo lista explicitamente por claridad
#===============================================================

def dividir_texto_en_frases_y_segmentos(texto: str, max_palabras: int = 80) -> list[str]:
    """
    Divide el texto en frases completas y luego agrupa esas frases
    en segmentos (párrafos técnicos) de hasta 'max_palabras' palabras.
    """
    import re

    # 1. Dividir en frases completas usando puntuación
    frases = re.split(r'(?<=[.!?])\s+', texto.strip())

    segmentos = []
    segmento_actual = []
    palabras_actual = 0

    # 2. Agrupar frases hasta llegar al límite de palabras
    for frase in frases:
        palabras_frase = len(frase.split())

        if palabras_actual + palabras_frase > max_palabras:
            if segmento_actual:
                segmentos.append(" ".join(segmento_actual))
            segmento_actual = [frase]
            palabras_actual = palabras_frase
        else:
            segmento_actual.append(frase)
            palabras_actual += palabras_frase

    # 3. Añadir el último segmento si quedó algo
    if segmento_actual:
        segmentos.append(" ".join(segmento_actual))

    return segmentos

#==================================
# 3. Genera la matriz de embeddings
#==================================


def generar_embeddings(chunks: list[str], modelo: SentenceTransformer) -> np.ndarray:
    """Convierte cada chunk en un embedding."""
    embeddings = modelo.encode(chunks)
    return np.array(embeddings)

 
 #==============================================
 #4. Encapsular todo en una función orquestadora
 #==============================================

def preparar_embeddings(texto: str) -> tuple[list[str], np.ndarray, SentenceTransformer]:
    """
    Divide el texto en chunks, carga el modelo y genera embeddings.
    Devuelve: (lista_chunks, matriz_embeddings, modelo)
    """

    try:
        print("Dividiendo texto en segmentos...")
        segmentos = dividir_texto_en_frases_y_segmentos(texto)
        print('\nCantidad de chunks: ', len(segmentos))
        input('Pulsa enter para continuar')

        print("Cargando modelo de embeddings...")
        modelo = cargar_modelo()

        print("Generando embeddings...")
        embeddings = generar_embeddings(segmentos, modelo)

        print("Embeddings generados correctamente.")
        return segmentos, embeddings, modelo
    
    except Exception as e:
        print(f'Error al preparar embeddings: {e}')
        return [], np.array([]), None



# ---------------------------------------------------------
# Prueba rápida del módulo
# ---------------------------------------------------------
if __name__ == "__main__":

    modelo = cargar_modelo()
    texto_prueba = "Este es un texto de prueba para verificar el módulo de embeddings. " * 50
    segmentos, emb, mod= preparar_embeddings(texto_prueba)

    print(f"Chunks generados: {len(segmentos)}")
    print(f"Dimensión de un embedding: {emb[0].shape}")
    for i, chunk in enumerate(segmentos):
        print(f"Chunk {i}: {chunk[:80]}...")
        print(f"Embedding: {emb[i][:3]}...\n")


