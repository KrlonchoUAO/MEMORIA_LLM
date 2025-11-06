# main_rag.py
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ------------------ Datos y Vector Store ------------------
docs = [
    "Riopaila Castilla es un conglomerado agroindustrial colombiano.",
    "La memoria a corto plazo en agentes guarda el historial de chat de la sesión.",
    "LangGraph usa checkpointers (MemorySaver) para persistir el estado por thread."
]
documents = [Document(page_content=t) for t in docs]

emb = OllamaEmbeddings(model="nomic-embed-text")  # usando embeddings locales con Ollama
vs = Chroma.from_documents(documents, embedding=emb, persist_directory=".chroma")
retriever = vs.as_retriever(search_kwargs={"k": 3})

def format_docs(ds):
    return "\n\n".join(d.page_content for d in ds)

# ------------------ LLM y Prompt ------------------
llm = ChatOllama(model="qwen3:4b", temperature=0.2, model_kwargs={"num_ctx": 1024})


# CAMBIA el prompt por este
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente que responde SOLO usando el contexto y el historial."),
    MessagesPlaceholder("chat_history"),
    ("human", "Pregunta: {question}\n\nContexto:\n{context}")
])


# CAMBIA la construcción de rag_chain por esta
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,  # <- solo la pregunta entra al retriever
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
)


# ------------------ Memoria de corto plazo (LangChain) ------------------
store = {}  # sesión -> historial

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rag_with_mem = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def run_demo(session_id: str):
    cfg = {"configurable": {"session_id": session_id}}

    print("\n--- Turno 1 (enseñando un dato) ---")
    r1 = rag_with_mem.invoke({"question": "Mi nombre es Momo. Recuérdalo."}, cfg)
    print(r1.content)

    print("\n--- Turno 2 (debe recordar) ---")
    r2 = rag_with_mem.invoke({"question": "¿Cómo me llamo?"}, cfg)
    print(r2.content)

if __name__ == "__main__":
    # Misma sesión: debe recordar
    run_demo("momo-001")

    # Sesión distinta: NO debe recordar
    print("\n======= Nueva sesión (sin memoria previa) =======")
    cfg2 = {"configurable": {"session_id": "momo-002"}}
    r3 = rag_with_mem.invoke({"question": "¿Cómo me llamo?"}, cfg2)
    print("\n--- Turno único en sesión nueva ---")
    print(r3.content)
