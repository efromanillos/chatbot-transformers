#==========================
#menu.py: menu para entrada
#de usuario
#==========================

import os

def menu():

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
                input("\nPulsa ENTER para continuar...")

            case '2': 
                print('Saliendo...')
                break
            case _:
                print('Opción NO válida')
                input("\nPulsa ENTER para continuar...")






if __name__ == '__main__':

    menu()
