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
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("Falta GROQ_API_KEY en el entorno. Añádelo a tu .env o variables de entorno.")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {DATA_DIR} (revisa que exista la carpeta 'Data')")

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
    system_prompt = """Eres un asesor de carreras de la Universidad Loyola.
Usa SOLO la información del contexto para responder.
Si el contexto tiene datos sobre una empresa, recomiéndala.

CONTEXTO:
{context}"""

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


def consultar(pregunta: str) -> str:
    global RAG_CHAIN
    if RAG_CHAIN is None:
        RAG_CHAIN = build_rag_chain()
    return RAG_CHAIN.invoke({"input": pregunta})


# ====== Gradio (lo más simple posible) ======
def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    return consultar(message)


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
        )

    demo.launch()


if __name__ == "__main__":
    main()