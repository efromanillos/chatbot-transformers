# Chatbot PLN con Transformers

Este proyecto forma parte de la actividad de Procesamiento del Lenguaje Natural (PLN).  
El objetivo es construir un chatbot capaz de responder preguntas basadas en un documento PDF, utilizando modelos del *stack* de **Hugging Face Transformers** (sin emplear LLMs grandes).

## Objetivo de la actividad

- Leer un documento PDF.
- Procesar su contenido.
- Generar **embeddings semánticos** para poder recuperar información relevante.
- Permitir que el usuario haga preguntas y obtener respuestas basadas en el contenido del PDF.
- Utilizar únicamente modelos que formen parte del ecosistema **Transformers**.

## Tecnologías utilizadas

- **Transformers** (Hugging Face)  
- **Torch** (backend para ejecutar modelos)  
- **Sentence-Transformers** (embeddings semánticos)  
- **PyPDF2** (lectura del PDF)  
- Python estándar para limpieza básica del texto
- Pipelines de Transformers (summarize y TTS)


## Funcionalidades implementadas (versión inicial)

- Carga y lectura del PDF.
- Limpieza básica del texto.
- Generación de embeddings del documento.
- Búsqueda semántica para recuperar los fragmentos más relevantes.
- Respuestas basadas en el contenido del documento.
- Detección automática del idioma del usuario.
- Chat con contexto activo e historial de sesión.
- Función resumen a través del comando: /resumen
- Función texto-to-speech de las respuesta y el resumen a través del comando: /voz
- Interfaz ASCII

## Instalación

Instalar dependencias principales:

- pip install transformers
- pip install torch
- pip install sentence-transformers
- pip install PyMuPDF
- pip install pygame
- pip install numpy


## ▶️ Ejecución

Desde el directorio raíz del programa:
    python main.py

## Mejoras futuras

- Añadir eliminación de *stopwords* con NLTK 
- Mejorar la gestión del contexto.
- Añadir análisis de sentimiento.
- Crear un pequeño front-end
- **langdetect** (detección de idioma del usuario)
- Multilingüe
- Persistencia del historial de conversación.

---


