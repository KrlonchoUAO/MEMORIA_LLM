# app.py
import os
from operator import itemgetter
from typing import Annotated, List

import streamlit as st

# ---- LangChain (LLM + Embeddings + VectorStore + Memoria) ----
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# ---- LangGraph (Estado + Checkpointer) ----
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# =========================
#  Streamlit UI
# =========================
st.set_page_config(page_title="RAG + Memoria (LangChain & LangGraph, Ollama)", page_icon="🧠")

# ===== Sidebar (reemplaza ese bloque) =====
with st.sidebar:
    st.header("⚙️ Config")

    # Modelos que ya tienes en este laptop (según tu 'ollama list')
    available_models = [
        "qwen3:14b-q4_K_M",     # rápido y buen razonamiento de memoria
        "qwen3:14b",            # más calidad, más VRAM
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.1:8b",
        "gpt-oss:20b",          # MUY pesado (útil si tienes VRAM y paciencia)
    ]
    selected_model = st.selectbox("LLM en Ollama", available_models, index=0)
    st.caption("Sugerencia: usa un *q4_K_M* para GPU mediana; el 'full' si tienes VRAM amplia.")

    backend = st.radio("Backend", ["LangChain", "LangGraph"], index=0)
    session_id = st.text_input("session_id / thread_id", value="demo-001")
    st.caption("Usa el mismo ID para que recuerde; cambia el ID para 'olvidar'.")
    st.divider()
    EMBED_MODEL = "nomic-embed-text"
    st.markdown(f"**Embeddings:** `{EMBED_MODEL}`")

# ===== Crea el LLM a partir de la selección (reemplaza tu bloque del LLM) =====
# LLM (Ollama local)
# Ajustes razonables para GPU: num_ctx moderado. Si tu GPU aprieta, añade "gpu_layers": 0 para CPU.
llm = ChatOllama(
    model=selected_model,
    temperature=0.2,
    model_kwargs={
        "num_ctx": 2048,   # si notas OOM, baja a 1024
        # "gpu_layers": -1, # deja que Ollama decida; si falla, prueba 0 para forzar CPU
        # "num_batch": 16,  # puedes subir si tienes VRAM
    },
)

# =========================
#  Config básica del demo
# =========================
DOCS = [
    "Riopaila Castilla es un conglomerado agroindustrial colombiano.",
    "La memoria a corto plazo en agentes guarda el historial de chat de la sesión.",
    "LangGraph usa checkpointers (MemorySaver) para persistir el estado por thread.",
]

EMBED_MODEL = "nomic-embed-text"   # embeddings locales vía Ollama


# =========================
#  Vector stores (aislados)
# =========================
def build_retriever(chroma_dir: str):
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vs = Chroma.from_documents(
        [Document(page_content=t) for t in DOCS],
        embedding=emb,
        persist_directory=chroma_dir,
    )
    return vs.as_retriever(search_kwargs={"k": 2})  # k pequeño para no “ahogar” el historial


retriever_chain = build_retriever(".chroma_ui_chain")
retriever_graph = build_retriever(".chroma_ui_graph")


# =========================
#  LangChain: RAG + Memoria
# =========================
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

prompt_chain = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un asistente con MEMORIA DE CONVERSACIÓN gestionada EXTERNAMENTE. "
        "NO debes decir que no puedes recordar. "
        "Tu única fuente de memoria es el historial (chat_history). "
        "Si el usuario declara información personal como 'Me llamo X', debes almacenarla "
        "en el historial (que ya es gestionado automáticamente), y luego usarla "
        "para responder futuras preguntas. "
        "NUNCA digas frases como 'no tengo información', 'no puedo recordarte', "
        "o 'no está en mi base de datos'. "
        "Tu regla es: SI EN EL HISTORIAL aparece información relevante, úsala. "
        "Si NO aparece, entonces responde normalmente usando el contexto."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "Pregunta: {question}\n\nContexto (solo si hace falta):\n{context}"),
])


rag_chain = (
    {
        "context": itemgetter("question") | retriever_chain | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt_chain
    | llm
)

# Memory store (por session_id)
# ---- reemplaza tu _store y _get_history por esto ----

# Mantener el store a través de reruns de Streamlit
if "lc_store" not in st.session_state:
    st.session_state.lc_store = {}  # dict[str, InMemoryChatMessageHistory]

def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    store = st.session_state.lc_store
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rag_with_mem = RunnableWithMessageHistory(
    rag_chain,
    _get_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# =========================
#  LangGraph: RAG + Memoria
# =========================
class State(dict):
    messages: Annotated[List[AnyMessage], add_messages]  # historial por thread

def rag_node(state: State):
    # última pregunta del usuario
    last_user = state["messages"][-1]
    q = last_user.content
    # contexto de RAG para esa pregunta
    ctx_docs = retriever_graph.invoke(q)
    ctx = format_docs(ctx_docs)

    messages = [
        SystemMessage(content=(
            "Prioriza el HISTORIAL del chat para responder a datos personales del usuario. "
            "Si el historial ya contiene la respuesta, respóndela directamente. "
            "Usa el CONTEXTO recuperado solo si aporta información externa. "
            "No cites ni repitas el contexto literalmente."
        )),
        *state["messages"],  # historial de corto plazo
        SystemMessage(content=f"Contexto recuperado (solo si hace falta):\n{ctx}"),
    ]
    ai = llm.invoke(messages)
    return {"messages": [ai]}

# justo antes de compilar el grafo
if "graph_checkpointer" not in st.session_state:
    st.session_state.graph_checkpointer = MemorySaver()

checkpointer = st.session_state.graph_checkpointer

# Compila una sola vez y reutiliza
if "app_graph" not in st.session_state:
    graph = StateGraph(State)
    graph.add_node("rag", rag_node)
    graph.add_edge(START, "rag")
    graph.add_edge("rag", END)
    st.session_state.app_graph = graph.compile(checkpointer=checkpointer)

app_graph = st.session_state.app_graph


st.title("🧠 RAG + Memoria con Ollama")

# Historial visual de la UI (solo para mostrar mensajes en pantalla)
if "chat" not in st.session_state:
    st.session_state.chat = []  # [(role, content)]

# Render del historial visual
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Escribe tu mensaje… (p.ej., 'Mi nombre es Momo. Recuérdalo.')")

if prompt:
    # pinta turno del usuario (solo UI)
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- Invocación 100% LLM-driven ----
    if backend == "LangChain":
        cfg = {"configurable": {"session_id": session_id}}
        result = rag_with_mem.invoke({"question": prompt}, cfg)
        answer = result.content if hasattr(result, "content") else str(result)
    else:
        cfg = {"configurable": {"thread_id": session_id}}
        out = app_graph.invoke({"messages": [HumanMessage(content=prompt)]}, cfg)
        answer = out["messages"][-1].content

    # pinta turno del asistente (solo UI)
    st.session_state.chat.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

# =========================
#  Panel de depuración
# =========================
with st.expander("🔎 Memoria (debug)"):
    st.write(f"Backend: {backend} · ID: {session_id}")
    if backend == "LangChain":
        hist = _get_history(session_id).messages
        st.write(f"Mensajes en memoria (LangChain): {len(hist)}")
        for m in hist:
            content = getattr(m, "content", "")
            prefix = f"- {m.type}: "
            st.write(prefix + (content if len(content) < 160 else content[:160] + "…"))

        if st.button("🧹 Borrar memoria (LangChain) de este ID"):
            st.session_state.lc_store.pop(session_id, None)
            st.success("Memoria LangChain borrada para este ID. Escribe de nuevo.")
    else:
        st.info(
            "LangGraph usa MemorySaver por thread_id. Para 'borrar', cambia de ID. "
            "Para producción, usa un checkpointer persistente (SQLite/Redis/Postgres)."
        )

st.caption("💡 Prueba: en el mismo ID escribe 'Mi nombre es Carlos David. Recuérdalo.' y luego '¿Cómo me llamo?'. "
           "Cambia el ID para que 'olvide'.")
