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

def dividir_en_chunks(texto: str, tamaño: int = 120) -> list[str]:
    """Divide el texto en fragmentos de 'tamaño' palabras."""
    palabras = texto.split()
   

    ## Se genera una lista de chunks, cada uno con hasta 'tamaño' palabras cada uno
    chunks = []
    for i in range(0, len(palabras), tamaño):
        chunk = " ".join(palabras[i:i+tamaño])
        chunks.append(chunk)

    return chunks

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
        print("Dividiendo texto en chunks...")
        chunks = dividir_en_chunks(texto)
        print('\nCantidad de chunks: ', len(chunks))

        print("Cargando modelo de embeddings...")
        modelo = cargar_modelo()

        print("Generando embeddings...")
        embeddings = generar_embeddings(chunks, modelo)

        print("Embeddings generados correctamente.")
        return chunks, embeddings, modelo
    
    except Exception as e:
        print(f'Error al preparar embeddings: {e}')
        return [], np.array([]), None



# ---------------------------------------------------------
# Prueba rápida del módulo
# ---------------------------------------------------------
if __name__ == "__main__":
    texto_prueba = "Este es un texto de prueba para verificar el módulo de embeddings. " * 50
    chunks, emb = preparar_embeddings(texto_prueba)

    print(f"Chunks generados: {len(chunks)}")
    print(f"Dimensión de un embedding: {emb[0].shape}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk[:80]}...")
        print(f"Embedding: {emb[i][:3]}...\n")


