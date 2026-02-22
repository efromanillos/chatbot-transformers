#=============================================================
#preparar_pdf.py
#Módulo donde de carga el pdf, se extrae el texto y se limpia
#preparado para el modelo de embeddings
#=============================================================



import fitz
import re


def cargar_texto(ruta_pdf: str) -> str:

    """Carga un pdf PyMuPDF y devulve todo su teto en un str"""

    try:
        documento = fitz.open(ruta_pdf)
        texto = ''
        for pagina in documento:
            texto += pagina.get_text()
        documento.close()
        return texto
    
    except Exception as e:
        print(f'error al cargar el PDF: {e}')
        return ''
    
def limpiar_texto(texto: str) -> str:
    """Limpia el texto eliminando saltos raros, espacios múltiples, etc."""

    # Eliminar saltos de línea repetidos
    texto = re.sub(r'\n+', '\n', texto)

    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto)

    # Quitar espacios al inicio y final
    texto = texto.strip()

    return texto

def preparar_texto(ruta_pdf: str) -> str:
    """Carga y limpia el PDF"""
    texto = cargar_texto(ruta_pdf)
    texto_limpiio = limpiar_texto(texto)
    return texto_limpiio


if __name__ == '__main__':

    ruta = '../Textos/Rodriguez-Cronologia-de-la-Inteligencia-Artificial.pdf'
    texto = preparar_texto(ruta)
    print(texto[:1000])
