#====================================
#menu.py: menu principal para entrada
#de usuario
#====================================


import os
from Codigo.preparar_pdf import cargar_texto, dividir_texto_en_frases_y_segmentos
from Codigo.embeddings import preparar_embeddings, MODELO_EMBEDDINGS
from Codigo.chatbot import chatear
from Codigo.utilidades import MODELO_RESUMEN, MODELO_TTS



#=================
# menu_principal()
#=================

def menu_principal():

    os.system('cls')
    print('\n\033[107;30m--< Iniciando Chatbot con Transformers >-- \033[0m\n')



    #Carga de datos necesarios para el chatbot:

    # 1. Declarar la ruta al pdf
    ruta_pdf = 'Textos/Rodriguez-Cronologia-de-la-Inteligencia-Artificial.pdf'

    # 2. Cargar pdf
    texto = cargar_texto(ruta_pdf)
   
    # 3.Obtener lista de segmentos del pdf
    print("Dividiendo texto en segmentos...")
    segmentos = dividir_texto_en_frases_y_segmentos(texto) 
    

    # # 4. Generar embeddings y cargar el modelo
    segmentos, embeddings, modelo = preparar_embeddings(segmentos)

    # print(f"· Modelo de resumen: {resumen_pipeline.model.name_or_path}")

    

    print("\033[107;30m------------------< Información del sistema >-----------------\033[0m\n")
    print(f"· Cantidad de segmentos: {len(segmentos)}")
    print(f"· Modelo de embeddings: {MODELO_EMBEDDINGS}")
    print(f"· Modelo de resumen: {MODELO_RESUMEN}")
    print(f"· Modelo de text-to-speech: {MODELO_TTS}")
    print("\n\033[107;30m--------------------------------------------------------------\033[0m")
    
    input('Pulsa ENTER para continuar...')

    # 4. Menú interactivo

    while True:

        os.system('cls')
        print('\033[107;30m--------< ChatBot con Transformers >---------\033[0m\n')
        print('1. Chatear')
        print('2. Salir')
        print('\n\033[107;30m----------------------------------------------\033[0m')

        opc = input('Seleccina una opción: ')

        match opc:

            case '1':
                print('...iniciando chatbot...')
                chatear(segmentos, embeddings, modelo)
                input("\nPulsa ENTER para continuar...")

            case '2': 
                print('Saliendo...')
                break
            case _:
                print('Opción NO válida')
                input("\nPulsa ENTER para continuar...")




# PARA PRUEBAS

if __name__ == '__main__':

    menu_principal()
