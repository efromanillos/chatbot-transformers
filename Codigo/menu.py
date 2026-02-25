#====================================
#menu.py: menu principal para entrada
#de usuario
#====================================


import os
from preparar_pdf import cargar_texto, dividir_texto_en_frases_y_segmentos
from embeddings import preparar_embeddings
from chatbot import chatear
from utilidades import resumen_pipeline



#=================
# menu_principal()
#=================

def menu_principal():

    os.system('cls')
    print('\n|Iniciando Chatbot con Transformers|\n')



    #Carga de datos necesarios para el chatbot:

    # 1. Declarar la ruta al pdf
    ruta_pdf = '../Textos/Rodriguez-Cronologia-de-la-Inteligencia-Artificial.pdf'

    # 2. Cargar pdf
    texto = cargar_texto(ruta_pdf)
   
    # 3.Obtener lista de segmentos del pdf
    print("Dividiendo texto en segmentos...")
    segmentos = dividir_texto_en_frases_y_segmentos(texto) 
    

    # # 4. Generar embeddings y cargar el modelo
    segmentos, embeddings, modelo = preparar_embeddings(segmentos)


    print("\033[92m------------------< Información del sistema >-----------------\033[0m")
    print(f"· Cantidad de segmentos: {len(segmentos)}")
    print(f"· Modelo de embeddings: {modelo._modules['0'].auto_model.config._name_or_path}")
    print(f"· Modelo de resumen: {resumen_pipeline.model.name_or_path}")
    print("\033[92m--------------------------------------------------------------\033[0m")
    
    input('Pulsa ENTER para continuar')

    # 4. Menú interactivo

    while True:

        os.system('cls')

        print('========= ChatBot con Transformers ==========')
        print('1. Chatear')
        print('2. Salir')
        print('=============================================')

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
