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

#===============================================================
# 2. Dividir texto en segmentos
# se especifica la salida tipo lista explicitamente por claridad
#===============================================================

def dividir_texto_en_frases_y_segmentos(texto: str, max_palabras: int = 80) -> list[str]:
    """
    Divide el texto en frases completas y luego agrupa esas frases
    en segmentos (párrafos técnicos) de hasta 'max_palabras' palabras.
    """

    # 0. Limpiar texto
    texto = limpiar_texto(texto)

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


"""
def preparar_texto(ruta_pdf: str) -> str:
    #Carga y limpia el PDF

    texto = cargar_texto(ruta_pdf)

    texto_limpio = limpiar_texto(texto)

    return texto_limpio

"""




if __name__ == '__main__':

    ruta = '../Textos/Rodriguez-Cronologia-de-la-Inteligencia-Artificial.pdf'
    texto = cargar_texto(ruta)
    texto_d = dividir_texto_en_frases_y_segmentos(texto)
    print(texto_d[:1000])
