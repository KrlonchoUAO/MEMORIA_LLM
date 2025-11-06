# main_graph.py
from typing import Annotated
from operator import itemgetter

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

# ------------------ Datos y Vector Store ------------------
docs = [
    "Riopaila Castilla es un conglomerado agroindustrial colombiano.",
    "La memoria a corto plazo en agentes guarda el historial de chat de la sesión.",
    "LangGraph usa checkpointers (MemorySaver) para persistir el estado por thread."
]
documents = [Document(page_content=t) for t in docs]

emb = OllamaEmbeddings(model="nomic-embed-text")
vs = Chroma.from_documents(documents, embedding=emb, persist_directory=".chroma_graph")
retriever = vs.as_retriever(search_kwargs={"k": 3})

def format_docs(ds):
    return "\n\n".join(d.page_content for d in ds)

# ------------------ Modelo (ajustado a tu RAM) ------------------
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.2,
    model_kwargs={"num_ctx": 1024}  # agrega "gpu_layers": 0 si necesitas CPU puro
)

# ------------------ Definir el estado del grafo ------------------
class State(dict):
    # Canal de historial: cada paso añade mensajes aquí
    messages: Annotated[list[AnyMessage], add_messages]

# ------------------ Nodo RAG ------------------
def rag_node(state: State):
    # 1) Tomamos la última pregunta del usuario
    last_user = state["messages"][-1]
    q = last_user.content

    # 2) Recuperamos contexto (RAG) usando SOLO la pregunta
    ctx_docs = retriever.invoke(q)
    ctx = format_docs(ctx_docs)

    # 3) Construimos los mensajes SIN duplicar el último human.
    #    Metemos el contexto como instrucción de sistema.
    messages = [
        SystemMessage(content="Eres un asistente que responde SOLO con el contexto y el historial. "
                              "Si el historial contiene el nombre del usuario o un dato pedido, respóndelo directamente."),
        *state["messages"],  # historial corto plazo (incluye el último HumanMessage con la pregunta)
        SystemMessage(content=f"Contexto recuperado (úsa SOLO esto):\n{ctx}")
    ]

    # 4) Llamada al LLM
    ai = llm.invoke(messages)

    # 5) Devolver el nuevo mensaje AI para que se agregue al historial
    return {"messages": [ai]}

# ------------------ Grafo ------------------
graph = StateGraph(State)
graph.add_node("rag", rag_node)
graph.add_edge(START, "rag")
graph.add_edge("rag", END)

# Checkpointer = memoria por thread_id (persistente mientras dure el proceso)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ------------------ Demo ------------------
def run_demo(thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    print("\n--- Turno 1 (enseñando un dato) ---")
    out1 = app.invoke({"messages": [HumanMessage(content="Mi nombre es Momo. Recuérdalo.")]}, cfg)
    print(out1["messages"][-1].content)

    print("\n--- Turno 2 (debe recordar) ---")
    out2 = app.invoke({"messages": [HumanMessage(content="¿Cómo me llamo?")]}, cfg)
    print(out2["messages"][-1].content)

if __name__ == "__main__":
    # Mismo thread: recuerda
    run_demo("thread-momo-001")

    # Thread distinto: no debería recordar
    print("\n======= Nuevo thread (sin memoria previa) =======")
    cfg2 = {"configurable": {"thread_id": "thread-momo-002"}}
    out3 = app.invoke({"messages": [HumanMessage(content="¿Cómo me llamo?")]}, cfg2)
    print("\n--- Turno único en thread nuevo ---")
    print(out3["messages"][-1].content)
