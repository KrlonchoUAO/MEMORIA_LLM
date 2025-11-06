import os
from operator import itemgetter
from typing import Annotated, List

import streamlit as st

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Startup: Data + Vector Store
# -----------------------------
DOCS = [
    "Riopaila Castilla es un conglomerado agroindustrial colombiano.",
    "La memoria a corto plazo en agentes guarda el historial de chat de la sesión.",
    "LangGraph usa checkpointers (MemorySaver) para persistir el estado por thread.",
]

EMBED_MODEL = "nomic-embed-text"  # embeddings locales vía Ollama
CHROMA_DIR_CHAIN = ".chroma_ui_chain"
CHROMA_DIR_GRAPH = ".chroma_ui_graph"

# Crea/recupera retriever para cada modo para evitar conflictos de persistencia
emb = OllamaEmbeddings(model=EMBED_MODEL)
vs_chain = Chroma.from_documents([Document(page_content=t) for t in DOCS], embedding=emb, persist_directory=CHROMA_DIR_CHAIN)
vs_graph = Chroma.from_documents([Document(page_content=t) for t in DOCS], embedding=emb, persist_directory=CHROMA_DIR_GRAPH)
retriever_chain = vs_chain.as_retriever(search_kwargs={"k": 3})
retriever_graph = vs_graph.as_retriever(search_kwargs={"k": 3})

# Modelo base (ajustado para equipos modestos)
LLM_MODEL = "qwen3:4b"
llm = ChatOllama(model=LLM_MODEL, temperature=0.2, model_kwargs={"num_ctx": 1024})

# -----------------------------
# LangChain path (RunnableWithMessageHistory)
# -----------------------------

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

prompt_chain = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente que responde SOLO usando el contexto y el historial. No repitas el contexto literalmente."),
    MessagesPlaceholder("chat_history"),
    ("human", "Pregunta: {question}\n\nContexto:\n{context}")
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

_store = {}

def _get_history(session_id: str):
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

rag_with_mem = RunnableWithMessageHistory(
    rag_chain,
    _get_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# -----------------------------
# LangGraph path (StateGraph + MemorySaver)
# -----------------------------

class State(dict):
    messages: Annotated[List[AnyMessage], add_messages]

def rag_node(state: State):
    # última pregunta
    last_user = state["messages"][-1]
    q = last_user.content
    ctx = format_docs(retriever_graph.invoke(q))

    messages = [
        SystemMessage(content=(
            "Eres un asistente que responde SOLO con el contexto y el historial. "
            "No cites ni repitas el contexto literalmente; responde de forma concisa."
        )),
        *state["messages"],
        SystemMessage(content=f"Contexto recuperado (ÚSALO SIN CITAR):\n{ctx}"),
    ]
    ai = llm.invoke(messages)
    return {"messages": [ai]}

_graph = StateGraph(State)
_graph.add_node("rag", rag_node)
_graph.add_edge(START, "rag")
_graph.add_edge("rag", END)

checkpointer = MemorySaver()
app_graph = _graph.compile(checkpointer=checkpointer)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG + Memoria (LangChain & LangGraph, Ollama)", page_icon="🧠")

with st.sidebar:
    st.header("⚙️ Config")
    backend = st.radio("Backend", ["LangChain", "LangGraph"], index=0)
    session_id = st.text_input("session_id / thread_id", value="demo-001")
    st.caption("Usa el mismo ID para que recuerde, cambia el ID para 'olvidar'.")
    st.divider()
    st.markdown("**Modelo:** `" + LLM_MODEL + "` · **Embeddings:** `" + EMBED_MODEL + "`")

st.title("🧠 RAG + Memoria con Ollama")

if "chat" not in st.session_state:
    st.session_state.chat = []  # [(role, content)]

# Mostrar historial
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Escribe tu mensaje… (p.ej., 'Mi nombre es Momo. Recuérdalo.')")

if prompt:
    # pinta turno del usuario
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # invocar backend
    if backend == "LangChain":
        cfg = {"configurable": {"session_id": session_id}}
        # LangChain espera question + (historial lo maneja internamente)
        result = rag_with_mem.invoke({"question": prompt}, cfg)
        answer = result.content if hasattr(result, "content") else str(result)

    else:  # LangGraph
        cfg = {"configurable": {"thread_id": session_id}}
        out = app_graph.invoke({"messages": [HumanMessage(content=prompt)]}, cfg)
        answer = out["messages"][-1].content

    # pinta turno del asistente
    st.session_state.chat.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

st.caption("💡 Prueba: en el mismo ID escribe 'Mi nombre es Momo. Recuérdalo.' y luego '¿Cómo me llamo?'. Cambia el ID para que 'olvide'.")
