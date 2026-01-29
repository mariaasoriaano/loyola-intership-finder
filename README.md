# Loyola Internship Finder – Proyecto RAG

## 1. Descripción general

Este proyecto implementa un **chatbot basado en Retrieval-Augmented Generation (RAG)** cuyo objetivo es ayudar a estudiantes de la Universidad Loyola a **identificar y comparar empresas donde realizar prácticas**, utilizando información empresarial real previamente estructurada.

El sistema combina:
- Documentación empresarial resumida en formato Markdown.
- Recuperación semántica mediante embeddings.
- Generación de respuestas con modelos de lenguaje, restringidas al contexto recuperado.

El proyecto tiene un **enfoque académico y demostrativo**, comparando una implementación RAG sencilla con una versión más avanzada que nos proporciona el profesor (“experta”).

---

## 2. Propósito del proyecto

Los objetivos principales son:

- Aplicar de forma práctica los conceptos de **RAG y sistemas conversacionales**.
- Evitar respuestas genéricas o inventadas mediante el uso explícito de contexto documental.
- Facilitar una primera orientación a estudiantes basada en datos empresariales reales.
- Comparar distintos niveles de complejidad en una arquitectura RAG.

El proyecto **no tiene finalidad comercial** ni sustituye la orientación académica personalizada.

---

## 3. Integrantes del equipo

- **Jesús Osorio Ortega**
- **María Soriano Flores**

Máster MBA + Data Analytics  
Universidad Loyola Andalucía

---

## 4. Tutor académico

- **Tutor:** David Carrascal
- **Asignatura / contexto:** Proyectos - Retrieval-Augmented Generation (RAG)

---

## 5. Funcionamiento general del sistema

El flujo del sistema es el siguiente:

1. Los datos empresariales originales se organizan por empresa.
2. Un script transforma estos datos en documentos Markdown resumidos.
3. Los documentos se convierten en embeddings y se almacenan en una base de datos vectorial.
4. Ante una consulta del usuario:
   - Se recupera el contexto más relevante.
   - El modelo de lenguaje genera una respuesta utilizando únicamente ese contexto.

---

## 6. Estructura del proyecto
```bash
loyola-intership-finder/
│
├── companies/ # Datos fuente organizados por empresa
│ ├── Alvaro_Moreno/
│ ├── Bida_Farma/
│ ├── Endesa/
│ ├── Grupo_Hermanos_Martin/
│ ├── Heineken/
│ ├── Migasa_Aceites/
│ └── Persan/
│
├── Data/ # Documentos Markdown finales usados por el RAG
│ ├── Alvaro_Moreno.md
│ ├── Bida_Farma.md
│ ├── Endesa.md
│ ├── Grupo_Hermanos_Martin.md
│ ├── Heineken.md
│ ├── Migasa_Aceites.md
│ └── Persan.md
│
├── chroma_db/ # Base de datos vectorial (RAG simple)
├── chroma_db_expert/ # Base de datos vectorial (RAG experto)
│
├── notebooks/ # Pruebas y experimentación en Jupyter
│
├── chat_interface.py # Implementación RAG simple + interfaz Gradio
├── app_expert.py # Implementación RAG avanzada
├── to_markdown.py # Conversión de datos a Markdown
├── rag_simple.ipynb # Ejemplo base de RAG proporcionado por el profesor
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env # Variables de entorno (no incluido)
```

---

## 7. Archivos y carpetas no incluidos en el repositorio

Por motivos de **privacidad, licencia y seguridad**, no se incluyen ni deben subirse:

- `companies/`  
  Contiene datos empresariales derivados de la plataforma SABI.

- `.env`  
  Contiene tokens y claves de acceso a servicios externos.

- `chroma_db/` y `chroma_db_expert/`  
  Bases de datos vectoriales generadas automáticamente durante la ejecución.

Estas carpetas se generan o configuran localmente.

---

## 8. Tecnologías utilizadas

- **Python**
- **LangChain**
- **ChromaDB**
- **Hugging Face** (embeddings y modelos)
- **Groq** (modelo de lenguaje)
- **Gradio** (interfaz web)
- **Jupyter Notebook**

---

## 9. Requisitos previos

- Python 3.9.6
- Entorno virtual recomendado
- Tokens de acceso a:
  - **Hugging Face**
  - **Groq**

---

## 10. Configuración de variables de entorno

Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:
```bash
HUGGINGFACEHUB_API_TOKEN
GROQ_API_KEY
```

Estas credenciales son necesarias para generar embeddings y respuestas del modelo de lenguaje.

---

## 11. Instalación

### Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate 
```

### Instalar dependencias
```bash
pip install -r requirements.txt
```

## 12. Ejecución del proyecto

### Ejecutar versión simple
```bash
python chat_interface.py
```

Lanza una interfaz web mediante Gradio accesible desde el navegador.

### Ejecutar versión experta
```bash
python app_expert.py
```

Incluye mejoras adicionales como chunking de documentos, caché por hash y uso de historial conversacional.


## 13. Versiones del sistema
- **RAG Simple (chat_interface.py)**
  - Arquitectura básica
  - Fácil de entender y estable
  - Orientada a demostración
- **RAG Experto (app_expert.py)**
  - Chunking de documentos
  - Caché inteligente de embeddings
  - Uso de historial conversacional
  - Recuperación más precisa del contexto

## 14. Limitaciones

- El sistema solo responde con información contenida en los documentos cargados.

- No evalúa la disponibilidad real de prácticas en las empresas.

- No sustituye la orientación académica personalizada.

## 15. Trabajo futuro

- Ampliar el número de empresas disponibles.

- Integrar perfiles de estudiantes para personalizar recomendaciones.

- Mejorar el ranking y la explicación de las recomendaciones.

- Despliegue en un entorno institucional.