# 🧠 Conversational RAG Assistant with PDF Uploads & Session History

An interactive chatbot application powered by Retrieval-Augmented Generation (RAG) that allows users to upload PDFs and ask questions about their content in a session-aware, chat-like interface — powered by LangChain, Streamlit, Groq LLMs, HuggingFace Embeddings, and Chroma vector store.

![App Screenshot](./assets/app_screenshot.jpg)

---

## ✨ Features

- ✅ Upload and parse one or more PDF files
- ✅ Chunk and embed documents using HuggingFace embeddings
- ✅ Persist vectors with Chroma DB for fast retrieval
- ✅ Reformulate queries contextually using session chat history
- ✅ Use RAG to generate answers via Groq-hosted LLM (Gemma-2-9b-it)
- ✅ Conversational UI with chat message bubbles using Streamlit
- ✅ Session-based memory using LangChain's `ChatMessageHistory`

---

## 🎓 How It Works

### Step 1: Upload PDFs
Users can upload one or more PDF files using the left sidebar. The uploaded PDFs are temporarily stored and processed.

### Step 2: Chunking & Text Splitting
The uploaded documents are split into smaller chunks using LangChain's `RecursiveCharacterTextSplitter` with a configurable chunk size and overlap.

### Step 3: Embedding with HuggingFace
Each document chunk is embedded using the HuggingFace embedding model `all-MiniLM-L6-v2`. This transforms text into numerical vectors.

### Step 4: Store in Chroma Vector Database
The vector embeddings are saved in a local persistent Chroma vector store for fast and efficient retrieval.

### Step 5: Retriever with History Awareness
LangChain's `create_history_aware_retriever` uses chat history and current user input to formulate better queries to retrieve relevant documents.

### Step 6: RAG Generation
The retrieved documents are passed into a `stuff_documents_chain`, combined with the original question and history, and passed to the LLM to generate a concise answer.

### Step 7: Chat UI with Session-Based History
The chat UI shows messages using `st.chat_message`, and session-specific memory is managed via LangChain’s `ChatMessageHistory`.

---

## 📊 Tech Stack

| Layer          | Tech Used |
|----------------|-----------|
| UI             | Streamlit |
| LLM            | [Groq API](https://groq.com/) using Gemma-2-9b-it |
| Embeddings     | HuggingFace `all-MiniLM-L6-v2` |
| RAG Framework  | LangChain |
| Vector Store   | Chroma (local persistent) |
| PDF Handling   | LangChain's `PyPDFLoader` |
| Memory         | LangChain `ChatMessageHistory` |
| Runtime Env    | Python 3.10+ & `venv` or `conda` |

---

## 📷 Demo & Screenshots

### App UI
![Chat Screenshot](./assets/chat_example.jpg)

### Video Demo
https://github.com/yourusername/conversational-rag-assistant/assets/demo_video.mp4  
*(Replace with actual media)*

---

## 🥛 Setup & Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/conversational-rag-assistant.git
cd conversational-rag-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` File

```env
GROQ_API=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```bash
.
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed)
├── chroma_db/            # Persistent vector DB
├── assets/
│   ├── app_screenshot.jpg
│   └── demo_video.mp4
```

---

## ✨ Example Use Case

- Upload research papers → Ask questions across multiple PDFs
- Session ID "project1" → Saves memory thread across page refresh
- Continue chatting, and answers improve with history!

---

## 🔐 API Notes

- **Groq API:** Used to run the `Gemma-2-9b-it` model for fast generation
- **HuggingFace Token:** Required for `all-MiniLM-L6-v2` embeddings

---

## 📣 Future Ideas

- Export chat as PDF or Markdown
- Support other file types (TXT, DOCX)
- Use OpenAI or Ollama for alternate LLMs
- Upload to Hugging Face Spaces or Streamlit Community Cloud

---

## 🧑‍💻 Author

**Your Name**  
[@yourgithub](https://github.com/yourusername)

---

## 📄 License

MIT License — free for personal and commercial use.

