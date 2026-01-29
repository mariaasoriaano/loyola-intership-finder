#!/usr/bin/env python3
"""
🎓 RAG Expert - Buscador de Prácticas Loyola (Versión Avanzada)
Interfaz web con Gradio para consultar empresas de prácticas
con mejoras RAG avanzadas: Hybrid Search, Reranking, Chunking Inteligente
"""

import os
import hashlib
import shutil
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Retrievers avanzados
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================

print("🚀 Iniciando sistema RAG Expert...")
print("=" * 60)

# Cargar variables de entorno
load_dotenv()
print("✅ Variables de entorno cargadas")

# Configuración de rutas
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db_expert"
HASH_FILE = "./chroma_db_expert/.doc_hash"

print(f"📂 Directorio de datos: {DATA_DIR}")
print(f"💾 Directorio ChromaDB: {CHROMA_DIR}")
print("=" * 60)

# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================


def calculate_documents_hash(documents):
    """
    Calcula un hash único basado en el contenido de todos los documentos.
    Se usa para detectar si los documentos han cambiado.
    """
    content_str = "".join(
        sorted([doc.page_content + str(doc.metadata) for doc in documents])
    )
    return hashlib.sha256(content_str.encode()).hexdigest()


def get_stored_hash():
    """Lee el hash almacenado de la última indexación."""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    return None


def store_hash(doc_hash):
    """Almacena el hash de los documentos indexados."""
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(doc_hash)


def format_docs_with_metadata(docs):
    """
    Formatea los documentos recuperados incluyendo metadatos.
    Cada documento se separa claramente con nombre de empresa.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        company_name = doc.metadata.get("company_name", "Desconocida")
        formatted.append(
            f"--- DOCUMENTO {i}: {company_name.upper()} ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def format_history_for_prompt(history):
    """
    Formatea el historial de conversación para incluirlo en el prompt.
    """
    if not history:
        return ""

    formatted_history = []
    for exchange in history[-3:]:  # Últimos 3 intercambios para no sobrecargar
        if isinstance(exchange, (list, tuple)) and len(exchange) == 2:
            user_msg, assistant_msg = exchange
            formatted_history.append(f"Usuario: {user_msg}")
            formatted_history.append(f"Asistente: {assistant_msg}")

    if formatted_history:
        return (
            "HISTORIAL DE CONVERSACIÓN RECIENTE:\n"
            + "\n".join(formatted_history)
            + "\n\n"
        )
    return ""


# ============================================================
# CARGA DE DOCUMENTOS
# ============================================================

print("\n🔄 Cargando documentos...")
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

# ============================================================
# CHUNKING INTELIGENTE
# ============================================================

print("\n🔄 Aplicando chunking inteligente...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

# Propagar metadatos a los chunks
for chunk in chunks:
    if "company_name" not in chunk.metadata:
        file_path = Path(chunk.metadata.get("source", ""))
        chunk.metadata["company_name"] = file_path.stem

print(f"✅ Chunking completado: {len(documents)} documentos → {len(chunks)} chunks")
print(
    f"   Tamaño promedio de chunk: {sum(len(c.page_content) for c in chunks) // len(chunks)} caracteres"
)

# ============================================================
# MODELO DE EMBEDDINGS MEJORADO
# ============================================================

print("\n🔄 Cargando modelo de embeddings mejorado...")
print("   (intfloat/multilingual-e5-large - esto puede tardar la primera vez)")

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("✅ Modelo de embeddings cargado (multilingual-e5-large)")

# ============================================================
# BASE DE DATOS VECTORIAL CON PERSISTENCIA INTELIGENTE
# ============================================================

print("\n🔄 Verificando base de datos vectorial...")

# Calcular hash actual de los documentos
current_hash = calculate_documents_hash(chunks)
stored_hash = get_stored_hash()

if stored_hash == current_hash and os.path.exists(CHROMA_DIR):
    # Los documentos no han cambiado, reutilizar la base de datos
    print("📦 Documentos sin cambios - Reutilizando base de datos existente...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR, embedding_function=embedding_model
    )
    print("✅ Base de datos vectorial cargada desde caché")
else:
    # Los documentos cambiaron o no existe la base de datos
    if os.path.exists(CHROMA_DIR):
        print("🗑️  Documentos modificados - Reconstruyendo base de datos...")
        shutil.rmtree(CHROMA_DIR)
    else:
        print("🆕 Creando nueva base de datos vectorial...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    store_hash(current_hash)
    print(f"✅ Base de datos vectorial creada con {len(chunks)} chunks")

# ============================================================
# CONFIGURACIÓN DEL LLM
# ============================================================

print("\n🔄 Configurando LLM...")
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,  # Más determinista para respuestas precisas
)
print("✅ LLM configurado (llama-3.3-70b-versatile)")

# ============================================================
# RETRIEVERS: HYBRID SEARCH + RERANKING
# ============================================================

print("\n🔄 Configurando sistema de búsqueda híbrida...")

# Retriever BM25 (búsqueda léxica)
print("   → Inicializando BM25 Retriever...")
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5  # Recuperar más documentos para el reranking

# Retriever Vectorial (búsqueda semántica)
print("   → Inicializando Vector Retriever...")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Ensemble Retriever (combinación híbrida)
print("   → Configurando Ensemble Retriever (BM25 40% + Vector 60%)...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
)
print("✅ Búsqueda híbrida configurada")

# ============================================================
# RERANKING CON CROSS-ENCODER
# ============================================================

print("\n🔄 Configurando Cross-Encoder Reranker...")
print("   (cross-encoder/ms-marco-MiniLM-L-6-v2)")

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=hybrid_retriever
)
print("✅ Reranking configurado (top 3 documentos más relevantes)")

# ============================================================
# PROMPT SYSTEM MEJORADO
# ============================================================

print("\n🔄 Configurando prompt avanzado...")

system_prompt = """Eres un asesor experto de carreras de la Universidad Loyola, especializado en ayudar a estudiantes a encontrar empresas de prácticas.

INSTRUCCIONES IMPORTANTES:
1. Usa ÚNICAMENTE la información proporcionada en el CONTEXTO para responder.
2. Si la información del contexto NO es suficiente o relevante para responder la pregunta, responde honestamente: "No tengo información suficiente sobre ese tema en mi base de datos de empresas de prácticas."
3. NO inventes información que no esté en el contexto.
4. Si mencionas una empresa, incluye detalles específicos del contexto (sector, ubicación, programas de prácticas, etc.).
5. Sé amable, profesional y orientado a ayudar al estudiante.

{history}CONTEXTO DE EMPRESAS DISPONIBLES:
{context}

FORMATO DE RESPUESTA:
- Sé conciso pero informativo
- Si recomiendas empresas, explica brevemente por qué son adecuadas
- Ofrece alternativas si es posible
- Menciona cualquier requisito o detalle relevante de las prácticas"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
print("✅ Prompt avanzado configurado")

# ============================================================
# CADENA RAG AVANZADA
# ============================================================

print("\n🔄 Construyendo cadena RAG avanzada...")


def create_rag_input(x):
    """Prepara el input para la cadena RAG con historial."""
    return {
        "input": x["input"],
        "history": x.get("history", ""),
    }


def retrieve_and_format(x):
    """Recupera documentos y los formatea con metadatos."""
    docs = reranking_retriever.invoke(x["input"])
    return format_docs_with_metadata(docs)


rag_chain = (
    RunnableParallel(
        {
            "context": RunnableLambda(lambda x: retrieve_and_format(x)),
            "input": lambda x: x["input"],
            "history": lambda x: x.get("history", ""),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Cadena RAG avanzada construida")

# ============================================================
# FUNCIÓN DE CHAT PARA GRADIO
# ============================================================


def chat(message, history):
    """
    Función de chat para la interfaz Gradio con soporte de historial.

    Args:
        message: El mensaje del usuario
        history: Historial de la conversación [(user, assistant), ...]

    Returns:
        La respuesta del sistema RAG avanzado
    """
    try:
        # Formatear historial para el contexto
        formatted_history = format_history_for_prompt(history)

        # Invocar la cadena RAG con el historial
        response = rag_chain.invoke({"input": message, "history": formatted_history})
        return response

    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return "⚠️ Se ha alcanzado el límite de consultas. Por favor, espera un momento e intenta de nuevo."
        return f"❌ Error al procesar la consulta: {error_msg}"


# ============================================================
# INTERFAZ GRADIO
# ============================================================

print("\n🔄 Creando interfaz Gradio...")

# Crear la interfaz de chat mejorada
demo = gr.ChatInterface(
    fn=chat,
    title="🎓 Buscador de Prácticas Loyola - Expert Edition",
    description="""
    **Sistema RAG Avanzado para buscar empresas de prácticas**
    
    ✨ **Mejoras implementadas:**
    - 🔍 **Búsqueda Híbrida**: Combina búsqueda semántica (embeddings) + léxica (BM25)
    - 🎯 **Reranking Inteligente**: Cross-Encoder para mayor precisión
    - 📝 **Chunking Optimizado**: Fragmentación inteligente de documentos
    - 🧠 **Embeddings Multilingües**: Modelo E5-large para mejor comprensión
    - 💬 **Memoria Conversacional**: Recuerda el contexto de la conversación
    - 💾 **Caché Inteligente**: Persistencia eficiente de la base de datos
    
    Haz preguntas sobre empresas disponibles para realizar prácticas.
    El sistema te recomendará empresas basándose en tus intereses y el contexto de la conversación.
    """,
    examples=[
        "Me gusta la cerveza, ¿qué empresa me recomiendas?",
        "Busco una empresa con buenos resultados financieros",
        "Mi madre es farmacéutica, ¿qué empresa me recomendarías?",
        "¿Qué empresas tienen programas de formación para becarios?",
        "Quiero hacer prácticas en el sector tecnológico",
    ],
    theme=gr.themes.Soft(),
    chatbot=gr.Chatbot(height=500),
)

print("✅ Interfaz Gradio creada")

# ============================================================
# LANZAR APLICACIÓN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎉 Sistema RAG Expert ready!")
    print("📊 Resumen de configuración:")
    print(f"   - Documentos originales: {len(documents)}")
    print(f"   - Chunks generados: {len(chunks)}")
    print(f"   - Modelo embeddings: multilingual-e5-large")
    print(f"   - Estrategia búsqueda: Hybrid (BM25 + Vector)")
    print(f"   - Reranking: Cross-Encoder (top 3)")
    print(f"   - LLM: llama-3.3-70b-versatile")
    print("🌐 Lanzando interfaz web...")
    print("=" * 60 + "\n")

    demo.launch()
