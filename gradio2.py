import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GROQ_API')
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize LLM and Embeddings
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# PDF processing functions
def pdfprocessing(pdf_file):
    documents = []
    for uploaded_file in pdf_file:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    return document_processing(documents)

def document_processing(documents):
    document_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    split_documents = document_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever()

# Chat session history handler
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Streamlit layout
st.set_page_config(layout="wide")
st.title("DocuMentor")

# Layout: Left for input, Right for chat
left_col, right_col = st.columns([1, 2], gap="large")

# LEFT COLUMN: Upload and session input
with left_col:
    st.header("📄 Upload & Session")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

# Load retriever with caching
@st.cache_resource(show_spinner=False)
def load_retriever(files):
    return pdfprocessing(files)

retriever = None
if uploaded_files:
    retriever = load_retriever(uploaded_files)

# Prompt setup
reformulate_prompt = (
    "You will be given Context, chat history and human input. "
    "You have to reformulate the input if needed as it will be used to retrieve docs. "
    "Only reformulate if needed else give as it is. Do not answer."
)

retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", reformulate_prompt),
    MessagesPlaceholder('Chat_history'),
    ("human", "{input}")
])

# RIGHT COLUMN: Chatbot
if retriever:
        history_plus_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        generative_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("Chat_history"),
            ("human", "{input}"),
        ])

        stuffing_docs = create_stuff_documents_chain(llm, generative_prompt)
        rag_chain = create_retrieval_chain(history_plus_retriever, stuffing_docs)

        conversational_chatbot = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key='Chat_history',
            output_messages_key="answer"
        )
        # Get the chat history for current session
        session_history = get_session_history(session_id)
        # Display previous chat messages
        with right_col:
            for msg in session_history.messages:
                with st.chat_message("user" if msg.type == "human" else "assistant"):
                    st.markdown(msg.content)

        # Chat input box at bottom
        user_input = st.chat_input("Ask something about your PDFs...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Thinking..."):
                response = conversational_chatbot.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            with st.chat_message("assistant"):
                st.markdown(response["answer"])
