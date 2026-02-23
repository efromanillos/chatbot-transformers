#====================================
#menu.py: menu principal para entrada
#de usuario
#====================================

import os
import os
from preparar_pdf import preparar_texto
from embeddings import preparar_embeddings, cargar_modelo
from chatbot import chatear


#=================
# menu_principal()
#=================

def menu_principal():

    #Carga de datos necesarios para el chatbot:

    # 1. Declarar la ruta al pdf
    ruta_pdf = '../Textos/Rodriguez-Cronologia-de-la-Inteligencia-Artificial.pdf'
   
    # 2.Obtener lista de segmentos del pdf
    segmentos = preparar_texto(ruta_pdf) 

    # 3. Obtener segmentos + embeddings + modelo
    segmentos, embeddings, modelo = preparar_embeddings(segmentos)

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
                #print("DEBUG modelo =", modelo)

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
