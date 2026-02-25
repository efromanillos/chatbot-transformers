#==============
#MÓDULO:
#utilidades.py
#==============

from transformers import pipeline
import os
import wave
import pygame
import numpy as np
import random

# Silenciar warnings molestos de HuggingFace, PyTorch y Python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

#=======
#MODELOS
#=======

MODELO_TTS = 'facebook/mms-tts-spa'
MODELO_RESUMEN = "sshleifer/distilbart-cnn-12-6"


# Cargamos el pipeline para resumir una sola vez
# NOTA: cuando se importa este módulo (utilidades.py) desde otro archivo,
# Python ejecuta todo el código de nivel superior (incluido este pipeline)
def cargar_resumen_pipeline():
    return pipeline("summarization", model=MODELO_RESUMEN)


# Cargar el pipeline de TTS en español

def cargar_tts_pipeline():
    return pipeline("text-to-speech", model=MODELO_TTS)



#===========================
#Genarar despedida aleatoria
#===========================

def generar_despedida():
    despedidas = [
        "Hasta otra!",
        "Que pases un buen día!",
        "Hasta la vista!",
        "Nos vemos pronto!",
        "Cuídate mucho!",
        "Fue un placer ayudarte!"
    ]
    return random.choice(despedidas)


#=======================
#Genarar resumen del pdf
#========================

def resumir_pdf(texto,  resumen_pipeline , max_length=60, min_length=30):

  
    # 1. Dividir el texto en fragmentos de máximo 900 caracteres
    # (para que no supere los 1024 tokens de entrada)
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

#===========================
#Genarar text-to-speech
#de la respuesta del chatbot
#===========================

def reproducir_voz(texto, tts_pipeline):

    audio = tts_pipeline(texto)
    raw = audio["audio"]
    sr = audio["sampling_rate"]

    # Interpretar PCM crudo como float32
    float_audio = np.frombuffer(raw, dtype=np.float32)

    # Convertir float32 (-1..1) a int16 (-32768..32767)
    int16_audio = (float_audio * 32767).astype(np.int16)

    
    #Apuntar al directorio temp, si no existe, se crea
    proyecto_dir = os.path.dirname(os.path.dirname(__file__))
    temp_dir = os.path.join(proyecto_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    wav_path = os.path.join(temp_dir, "temp_audio.wav")

    # Crear WAV válido PCM 16-bit
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16 bits
        wf.setframerate(sr)         # 16000 Hz
        wf.writeframes(int16_audio.tobytes())

    # Reproducir con pygame
    pygame.mixer.init()
    pygame.mixer.music.load(wav_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()  # liberar archivo

    os.remove(wav_path)  # ahora sí se puede borrar

