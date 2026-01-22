import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
print("✅ Dependencias cargadas")

============================================================================
============================================================================

# Configuración
DATA_DIR = "../data"
CHROMA_DIR = "../chroma_db"

# IMPORTANTE: Limpiar base de datos anterior para evitar duplicados
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print(f"🗑️  Base de datos anterior eliminada")

print(f"📂 Directorio de datos: {DATA_DIR}")
print(f"💾 Directorio ChromaDB: {CHROMA_DIR}")

============================================================================
============================================================================

# Cargar documentos COMPLETOS (sin chunking)
loader = DirectoryLoader(
    DATA_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

# Agregar metadata
for doc in documents:
    file_path = Path(doc.metadata.get("source", ""))
    doc.metadata["company_name"] = file_path.stem

print(f"✅ Documentos cargados: {len(documents)}")
for doc in documents:
    print(f"   - {doc.metadata['company_name']}: {len(doc.page_content)} caracteres")

============================================================================
============================================================================

# Modelo de Embeddings
print("🔄 Cargando modelo de embeddings...")
embedding_model = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
print("✅ Modelo cargado")

============================================================================
============================================================================

# Crear base de datos vectorial CON DOCUMENTOS COMPLETOS
print("🔄 Creando base de datos vectorial...")
vectorstore = Chroma.from_documents(
    documents=documents,  # Sin chunking - documentos completos
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
)
print(f"✅ Base de datos creada con {len(documents)} documentos")

============================================================================
============================================================================

# Configurar LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile"
)
print("✅ LLM configurado")

============================================================================
============================================================================

# Retriever simple
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Solo 1 documento
print("✅ Retriever configurado")

============================================================================
============================================================================

# Prompt simple
system_prompt = """Eres un asesor de carreras de la Universidad Loyola.
Usa SOLO la información del contexto para responder.
Si el contexto tiene datos sobre una empresa, recomiéndala.

CONTEXTO:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
print("✅ Prompt configurado")

============================================================================
============================================================================

# Cadena RAG
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
print("✅ Cadena RAG construida")

============================================================================
============================================================================

# Función simple para consultas
def consultar(pregunta):
    return rag_chain.invoke({"input": pregunta})


# Ejemplo de uso:
consultar("Mi madre es farmaceútica, ¿qué empresa me recomiendas?")S

============================================================================
============================================================================
