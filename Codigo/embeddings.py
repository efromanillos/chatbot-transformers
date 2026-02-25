#====================================================================
#emebddings.py
#Módulo que: 
# 1. Carga el modelo de embeddings
# 2. Generar la matrziz de embeddings
# 3. Encapsula todo en una función orquestadora preparar_emebddings()
#====================================================================

from sentence_transformers import SentenceTransformer
import numpy as np

#=====================
# MODELO DE EMBEDDINGS
#=====================

MODELO_EMBEDDINGS = 'all-MiniLM-L6-v2'


#===============================
# 1. Cargar modelo de embeddings
#===============================

def cargar_modelo() -> SentenceTransformer:
    """Cargar el modelo de embeddings solo una vez"""
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    return modelo


#==================================
# 2. Genera la matriz de embeddings
#==================================


def generar_embeddings(segmentos: list[str], modelo: SentenceTransformer) -> np.ndarray:
    """Convierte cada segmento en un embedding."""
    embeddings = modelo.encode(segmentos)
    return np.array(embeddings)

 
 #==============================================
 #3. Ejecutar todo en una función orquestadora
 #==============================================

def preparar_embeddings(segmentos: list[str]) -> tuple[list[str], np.ndarray, SentenceTransformer]:
    """
    Carga el modelo y genera embeddings.
    Devuelve: (lista_segmentos, matriz_embeddings, modelo)
    """

    try:
        
        print("Cargando modelo de embeddings...\n")
        modelo = cargar_modelo()
        

        print("\nGenerando embeddings...\n")
        embeddings = generar_embeddings(segmentos, modelo)

        print("Embeddings generados correctamente.\n")
        
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

    print(f"segmentos generados: {len(segmentos)}")
    print(f"Dimensión de un embedding: {emb[0].shape}")
    for i, chunk in enumerate(segmentos):
        print(f"Chunk {i}: {chunk[:80]}...")
        print(f"Embedding: {emb[i][:3]}...\n")


