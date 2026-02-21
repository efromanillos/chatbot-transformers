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
- **langdetect** (detección de idioma del usuario)  
- Python estándar para limpieza básica del texto

> Nota: De momento no se utiliza NLTK para mantener el proyecto simple.  
> Se podrá añadir más adelante si se desea mejorar la precisión eliminando *stopwords*.

## Funcionalidades implementadas (versión inicial)

- Carga y lectura del PDF.
- Limpieza básica del texto.
- Generación de embeddings del documento.
- Búsqueda semántica para recuperar los fragmentos más relevantes.
- Respuestas basadas en el contenido del documento.
- Detección automática del idioma del usuario.
- Chat básico con contexto limitado.

## Instalación

Instalar dependencias principales:

pip install transformers torch sentence-transformers PyPDF2 langdetect


## ▶️ Ejecución

(Se completará más adelante cuando esté listo el `main.py`.)

## Mejoras futuras

- Añadir eliminación de *stopwords* con NLTK 
- Mejorar la gestión del contexto.
- Añadir análisis de sentimiento.
- Crear un pequeño front-end.
- Persistencia del historial de conversación.

---


