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
import Codigo.historial as historial

from Codigo.utilidades import generar_despedida, resumir_pdf, reproducir_voz, cargar_tts_pipeline, cargar_resumen_pipeline

despedidas = [
    "Hasta otra.",
    "Que pases un buen día.",
    "Hasta la vista.",
    "Nos vemos pronto.",
    "Cuídate mucho.",
    "Fue un placer ayudarte."
]

#=================================================================
# buscar_similares()
# buscar imilitud entre pregunta y segmentos, devulve lista de tuplas
# con los 3 segmentos más similares
#=================================================================

def buscar_similares(pregunta: str, segmentos: list[str], embeddings: np.ndarray, modelo: SentenceTransformer, k: int = 3, umbral: float = 0.55) -> list[tuple[str,float]]:
    """Devuelve los k segmentos más similares a la pregunta en una lista de tuplas (chunk, similitudes)."""
    
    
    # Normalizar y reforzar la pregunta para mejorar la precisión, añade contexto a la pregunta
    pregunta_normalizada = pregunta.lower().strip().replace('?', '')
    pregunta_reforzada = f'Información sobre: {pregunta_normalizada}'
    
    
    # Embedding de la pregunta
    emb_pregunta = modelo.encode([pregunta_reforzada])

    # Similaridad coseno
    similitudes = cosine_similarity(emb_pregunta, embeddings)[0]

    # Ordena los índices según la similitud (de menor a mayor),
    # luego los invierte [::-1] para obtenerlos de mayor a menor
    # y finalmente se queda con los k primeros [:k], es decir,
    # los 3 índices con mayor similitud.

    indices = np.argsort(similitudes)[::-1][:k]

    # Mejor similitud
    mejor_similitud = similitudes[indices[0]]

    # Si no alcanza el umbral, no devolvemos nada
    if mejor_similitud < umbral:
        return []

    # Devolver lista de (chunk, similitud)
    resultado = []
    for i in indices:
        chunk = segmentos[i]
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

    # Tomamos el segmento y simmilitud más relevante
    mejor_segmento, mejor_similitud = contexto[0]

    # Respuesta simple
    respuesta = (
        'He encontrado la siguiente información relevante en el documento:\n\n'
        f'{mejor_segmento}\n\n'
        '---------------------------------------------------------------------'
    )

    return respuesta

#=====================================================
# responder()
# Función orquestadora:
# Busca los 3 segmentos más similares
# Devuelve un único chunk, el más relevante.
#=====================================================


def responder(pregunta: str, segmentos: list[str], embeddings: np.ndarray, modelo: SentenceTransformer) -> str:
    """
    Orquesta el proceso completo:
    1. Busca los segmentos más similares
    2. Genera una respuesta breve y relevante
    """

    # 1. Recuperar los segmentos más parecidos
    
    segmentos_similares = buscar_similares(pregunta, segmentos, embeddings, modelo, k=3, umbral = 0.50)

    # 2. Si no hay nada relevante, fallback response
    if not segmentos_similares:
        return "No he encontrado información relevante sobre eso en el PDF."

    # 3. Generar respuesta breve basada en el mejor chunk
    respuesta = generar_respuesta(pregunta, segmentos_similares)


    return respuesta

#==================
#mostrar_comandos() 
#==================
def mostrar_comandos():
    #print("\n\033[45m/voz   /resumen   /salir\033[0m\n")
    #print("\033[42m  COMANDOS:  /voz   /resumen   /salir  \033[0m")
    #print("\033[44m| COMANDOS |  /voz   /resumen   /salir\033[0m")
    print('\033[107;30m COMANDOS →  /voz   /resumen   /salir  \033[0m\n')
    #print('\033[100;97m COMANDOS → /voz /resumen /salir \033[0m')





#============================================
#chatear() 
#función para generar el bucle conversacional
#============================================


def chatear(segmentos, embeddings, modelo):

    print('\n === MODO CHAT ===\n')
    print('\033[92m <Chatbot>:\033[0m Hola, soy tu asistente. Pregúntame lo que quieras sobre el PDF. \n')
    print(' · Para realizar un resumen del texto escribe:\033[107;30m/resumen\033[0m')
    print(' · Para que el chatbot lea la respuesta en voz alta:\033[107;30m/voz\033[0m')
    print(' · Para salir del chat escribe:\033[107;30m/salir\033[0m\n')

    ultima_respuesta = None
    
    while True:
        
        pregunta = input('\033[91m <Usuario>: \033[0m')

        
        comando = pregunta.lower().strip()

        match comando:

            case '/salir':
                despedida = generar_despedida()
                print(f'\n\033[92m <Chatbot>: \033[0m {despedida}')
                print("\n Saliendo del chat...")
                break

            case '/resumen':
                texto_completo = " ".join(segmentos)
                resumen_pipeline = cargar_resumen_pipeline()
                resumen = resumir_pdf(texto_completo, resumen_pipeline)
                print(f'\n\033[92m <Chatbot>: \033[0mAquí tienes un resumen del documento:\n\n{resumen}\n')
                ultima_respuesta = resumen
                mostrar_comandos()
                continue

            case '/voz':
                if ultima_respuesta is None:
                    print('\n\033[92m <Chatbot>: \033[0mNo tengo ninguna respuesta previa para convertir a voz.\n')
                else:
                    print('\n\033[92m <Chatbot>: \033[0mGenerando voz en español...\n')
                    tts_pipeline = cargar_tts_pipeline()
                    reproducir_voz(ultima_respuesta, tts_pipeline)
                continue

            case _:
                # Aquí sigue el flujo normal del chatbot
                pass

        # 1. Detectar si la pregunta depende del contexto
        dependiente = historial.es_pregunta_dependiente(pregunta)

        # 2. Construir la consulta adecuada
        if dependiente:
            consulta = historial.construir_consulta_con_contexto(pregunta)
        else:
            consulta = pregunta

        # 3. Obtener respuesta usando la consulta contextual
        respuesta = responder(consulta, segmentos, embeddings, modelo)

        # 4. Guardar turno real (pregunta original + respuesta + si era dependiente)
        historial.agregar_turno(pregunta, respuesta, dependiente)

        print(f'\n\033[92m <Chatbot>: \033[0m{respuesta}\n')
        mostrar_comandos()
        ultima_respuesta = respuesta