import os
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


# =========================
# Config (muy parecido a tu .py original)
# Ajustado a tu estructura real (VS Code):
# loyola-internship-finder/
#   ├─ Data/        (markdowns)
#   ├─ chroma_db/   (Chroma)
#   ├─ chat_interface.py
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDINGS_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 3


def format_docs(docs):
    blocks = []
    for doc in docs:
        company = doc.metadata.get("company_name", "Empresa desconocida")
        blocks.append(f"EMPRESA: {company}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_rag_chain():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError(
            "Falta GROQ_API_KEY en el entorno. Añádelo a tu .env o variables de entorno."
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Directory not found: {DATA_DIR} (revisa que exista la carpeta 'Data')"
        )

    # Cargar documentos (.md)
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    # Metadata company_name
    for doc in documents:
        file_path = Path(doc.metadata.get("source", ""))
        doc.metadata["company_name"] = file_path.stem

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Vectorstore (reusar si existe, crear si no)
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding_model,
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=str(CHROMA_DIR),
        )

    # LLM
    llm = ChatGroq(groq_api_key=groq_key, model_name=GROQ_MODEL_NAME)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Prompt (igual idea que tu original)
    system_prompt = (
        system_prompt
    ) = """Eres un orientador profesional de la Universidad Loyola especializado en prácticas universitarias.

Tu tarea es analizar la información del CONTEXTO y recomendar empresas
que encajen con los intereses del estudiante.

REGLAS ESTRICTAS:
- Usa EXCLUSIVAMENTE la información del CONTEXTO
- No inventes datos ni hagas suposiciones externas
- Solo analiza empresas que aparezcan explícitamente en el contexto
- Si falta información para alguna parte del análisis, indícalo claramente

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA:

Empresa TOP recomendada: 

========================
RANKING DE EMPRESAS
========================

1. Empresa: <nombre>
   Nota sobre 10: scoring del 1 al 10
   Justificación detallada:
   <explicación brevey razonada de por qué esta empresa encaja con el perfil del estudiante,
   citando elementos concretos del contexto (sector, actividad, valores, tipo de negocio, etc.)>

2. Empresa: ...

========================
COMPARATIVA ECONÓMICA
========================

Realiza una comparación breve de la situación económica de las empresas recomendadas
utilizando únicamente la información disponible en el CONTEXTO.

Para cada empresa analiza, cuando esté disponible:
- Tamaño o dimensión de la empresa
- Estabilidad o solidez económica
- Sector y posición en el mercado
- Cualquier indicador económico mencionado explícitamente

Si el contexto no proporciona información económica suficiente para alguna empresa,
indícalo claramente sin inferir datos externos.

CONTEXTO:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # Cadena RAG
    rag_chain = (
        RunnableParallel(
            {
                "context": (lambda x: x["input"]) | retriever | format_docs,
                "input": lambda x: x["input"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Cadena global (se construye una vez)
RAG_CHAIN = None
ULTIMA_RESPUESTA = ""


def consultar(pregunta: str) -> str:
    global RAG_CHAIN
    if RAG_CHAIN is None:
        RAG_CHAIN = build_rag_chain()
    return RAG_CHAIN.invoke({"input": pregunta})


def chat_fn(message, history):
    """Función del chat que guarda la última respuesta."""
    global ULTIMA_RESPUESTA
    res = consultar(message)
    ULTIMA_RESPUESTA = res.split("\n")[0]
    print("EMPRESA ELEGIDA: " + ULTIMA_RESPUESTA)
    return res


def generate_email_fn():
    """Genera email basado en la última respuesta del chat."""
    global ULTIMA_RESPUESTA
    if not ULTIMA_RESPUESTA:
        return "Primero consulta información sobre una empresa en el chat."

    load_dotenv()
    chat = ChatGroq(temperature=0, model_name=GROQ_MODEL_NAME)

    res = chat.invoke(f"""
    Actúa como un experto copywritter y crea un email para pedirle a esta empresa hacer las prácticas como becario.

    Información de la empresa:
    {ULTIMA_RESPUESTA}
""")

    return res.content


def main():
    with gr.Blocks(title="Loyola Career Advisor") as demo:
        gr.Markdown(
            """
# **Loyola Career Advisor**

Descubre qué empresas encajan mejor contigo para hacer prácticas universitarias.

Esta herramienta te ayuda a orientarte según **tus intereses, lo que te gusta y el tipo de empresa que buscas**.
No necesitas saber nombres de empresas ni tenerlo claro desde el principio: basta con que nos cuentes
qué estudias, qué te motiva o qué tipo de entorno te atrae.

A partir de esa información, te recomendaremos empresas donde podrías encajar mejor para realizar tus
prácticas, basándonos en información real y actual.
"""
        )

        gr.ChatInterface(
            fn=chat_fn,
            examples=[
                "Me gusta la cerveza, ¿qué empresa me recomiendas?",
                "Busco prácticas en consultoría, ¿qué empresa encaja mejor?",
                "Me interesa energía y sostenibilidad, ¿qué empresa me recomiendas?",
            ],
            chatbot=gr.Chatbot(height=600),
            theme=gr.themes.Soft(),
        )

        gr.Markdown("---")
        gr.Markdown("### Generar email de solicitud de prácticas")

        boton_email = gr.Button("Generar Email")
        texto_email = gr.Textbox(label="Cuerpo del email", lines=10)
        boton_email.click(
            fn=generate_email_fn,
            inputs=[],
            outputs=texto_email,
        )

        copy_btn = gr.Button("📋 Copiar Texto")
        copy_btn.click(
            fn=None, inputs=[texto_email], js="(x) => navigator.clipboard.writeText(x)"
        )

    demo.launch()


if __name__ == "__main__":
    main()
