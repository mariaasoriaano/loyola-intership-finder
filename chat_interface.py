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
    """
    Construye el vectorstore y la cadena RAG.
    - Reutiliza chroma_db si existe y tiene contenido.
    - Si no existe (o está vacío), crea la BD a partir de Data/*.md
    """
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("Falta GROQ_API_KEY. Ponla en tu .env o como variable de entorno.")

    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de datos: {DATA_DIR}\n"
            f"Según tu estructura, debe ser: {BASE_DIR / 'Data'}"
        )

    # Cargar documentos
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        raise RuntimeError(f"No se han encontrado .md dentro de {DATA_DIR} (glob **/*.md).")

    # Añadir metadata company_name = nombre del archivo sin extensión
    for doc in documents:
        file_path = Path(doc.metadata.get("source", ""))
        doc.metadata["company_name"] = file_path.stem

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Vectorstore (reusar o crear)
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

    # Prompt
    system_prompt = """Eres un asesor de carreras de la Universidad Loyola.
Usa SOLO la información del contexto para responder.
Si el contexto tiene datos sobre una empresa, recomiéndala.

CONTEXTO:
{context}"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # RAG chain
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

    return rag_chain, len(documents)


# ==============
# Estado global
# ==============
RAG_CHAIN = None
DOC_COUNT = 0


def init_app():
    global RAG_CHAIN, DOC_COUNT
    RAG_CHAIN, DOC_COUNT = build_rag_chain()
    return f"✅ RAG listo. Documentos cargados: {DOC_COUNT}\n📁 Data: {DATA_DIR}\n🗄️ Chroma: {CHROMA_DIR}"


def consultar(pregunta: str) -> str:
    if RAG_CHAIN is None:
        init_app()
    return RAG_CHAIN.invoke({"input": pregunta})


def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    try:
        return consultar(message)
    except Exception as e:
        return f"❌ Error: {e}"


def status_action() -> str:
    if RAG_CHAIN is None:
        return f"ℹ️ RAG no inicializado todavía.\n📁 Data: {DATA_DIR}\n🗄️ Chroma: {CHROMA_DIR}"
    return f"✅ RAG activo. Documentos cargados: {DOC_COUNT}\n📁 Data: {DATA_DIR}\n🗄️ Chroma: {CHROMA_DIR}"


def main():
    # Inicializa al arrancar (sin rebuild)
    startup_msg = init_app()

    with gr.Blocks(title="Loyola Career Advisor (RAG)") as demo:
        gr.Markdown(
            "# Loyola Career Advisor (RAG)\n"
            f"Lee tus markdowns de `Data/` y responde con recuperación (Chroma).\n"
        )

        status = gr.Textbox(
            label="Estado",
            value=startup_msg,
            interactive=False,
            lines=3,
        )

        btn_status = gr.Button("Ver estado")
        btn_status.click(fn=status_action, outputs=status)

        gr.Markdown("## Chat")
        gr.ChatInterface(
            fn=chat_fn,
            examples=[
                "Mi madre es farmacéutica, ¿qué empresa me recomiendas?",
                "Busco prácticas en consultoría, ¿qué empresa encaja mejor?",
                "¿Qué empresas tienen un perfil más tecnológico?",
            ],
            retry_btn="Reintentar",
            undo_btn="Deshacer último",
            clear_btn="Borrar chat",
        )

    demo.launch()


if __name__ == "__main__":
    main()
