#==============
#MÓDULO:
#utilidades.py
#==============

from transformers import pipeline


# Cargamos el pipeline para resumir una sola vez
# NOTA: cuando se importa este módulo (utilidades.py) desde otro archivo,
# Python ejecuta todo el código de nivel superior (incluido este pipeline)
resumen_pipeline = pipeline("summarization")




#===========================
#Genarar despedida aleatoria
#===========================

import random

def generar_despedida():
    despedidas = [
        "Hasta otra",
        "Que pases un buen día",
        "Hasta la vista",
        "Nos vemos pronto",
        "Cuídate mucho",
        "Fue un placer ayudarte"
    ]
    return random.choice(despedidas)


#=======================
#Genarar resumen del pdf
#========================


from transformers import pipeline

resumen_pipeline = pipeline("summarization")

def resumir_pdf(texto, max_length=130, min_length=30):
    # 1. Dividir el texto en fragmentos de máximo 900 caracteres
    # (para que no supere los 1024 tokens)
    trozos = []
    paso = 900
    for i in range(0, len(texto), paso):
        trozos.append(texto[i:i+paso])

    # 2. Resumir cada fragmento por separado
    resumenes = []
    for t in trozos:
        r = resumen_pipeline(
            t,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        resumenes.append(r[0]["summary_text"])

    # 3. Unir los resúmenes en un único texto
    resumen_final = " ".join(resumenes)
    return resumen_final

