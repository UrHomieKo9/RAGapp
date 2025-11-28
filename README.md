# LangChain RAG Application

Retrieval-Augmented Generation stack built with FastAPI, LangChain, FAISS, and Streamlit. Upload PDFs, build embeddings, and chat with your documents through an intuitive UI.

## ⚙️ Architecture

```
                  +----------------------+
                  |      Streamlit       |
                  |  frontend/app.py     |
                  +----------+-----------+
                             |
                HTTP (upload/query/embed)
                             |
+----------------------------v-----------------------------+
|                        FastAPI                           |
|                    backend/main.py                       |
|  +------------------------+----------------------------+ |
|  |    RAGPipeline (LangChain)                          | |
|  |  - PDF Loading (PyPDFLoader)                        | |
|  |  - Chunking (RecursiveCharacterTextSplitter)        | |
|  |  - Embeddings (OpenAI / SentenceTransformer)        | |
|  |  - FAISS Vector Store                               | |
|  |  - Retriever + Chat LLM + Memory                    | |
|  +-----------------------------------------------------+ |
|                 Stores chunks/indices in /models          |
+-----------------------------------------------------------+
                          |
                        PDFs
                          |
                   data/sample_pdfs
```

## 📂 Project Layout

```
rag_app/
├── backend/
│   ├── main.py             # FastAPI server and endpoints
│   ├── rag_pipeline.py     # LangChain RAG orchestration
│   ├── config.py           # Settings via pydantic BaseSettings
│   └── utils.py            # Upload helpers
├── frontend/
│   └── app.py              # Streamlit UI
├── data/sample_pdfs/       # Drop your PDFs here (optional)
├── models/                 # Vector store artifacts (runtime)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

1. **Clone & install dependencies**

```bash
cd rag_app
python -m venv .venv && .venv\Scripts\activate  # Windows
# or: source .venv/bin/activate                # macOS/Linux
pip install -r requirements.txt
```

2. **Configure environment**

Create `.env` in `rag_app/`:

```
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai        # or groq (requires GROQ_API_KEY)
LLM_MODEL_NAME=gpt-4o-mini
```

Optional: `EMBEDDING_MODEL_NAME`, `CHUNK_SIZE`, `TOP_K`, etc. (see `backend/config.py`).

## 🖥️ Running the Services

### Backend (FastAPI)

```bash
cd rag_app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- `POST /upload` – upload PDF, build FAISS index
- `POST /embed` – embed ad-hoc text list
- `POST /query` – ask question, returns answer + sources + history
- `GET /health` – service health

### Frontend (Streamlit)

```bash
cd rag_app/frontend
streamlit run app.py
```

Update backend URL in the sidebar if not `http://localhost:8000`.

## 💡 Example Workflow

1. Drop one or more PDFs into `data/sample_pdfs/` (optional for convenience).
2. In Streamlit:
   - Upload a PDF (`Upload & Index`).
   - Type “What are the main findings?” and click **Ask**.
   - Review the answer plus cited document snippets.

Example questions:

- “Summarize the executive overview.”
- “List all deadlines mentioned.”
- “Which sections talk about security controls?”

## ⚠️ Limitations

- Relies on OpenAI (or Groq) for generation; requires API keys.
- In-memory FAISS index—no persistence between restarts unless extended.
- Single-document pipeline per session; multi-user/session routing not included.
- No authentication layers; place behind VPN/reverse proxy for production use.

## 🔭 Future Improvements

- Persist vector stores per document/user.
- Add background jobs for heavy ingestion pipelines.
- Support multi-file batch ingestion and deletion.
- Add observability (tracing, logging to OpenTelemetry).
- Plug in Guardrails for prompt safety and PII scrubbing.

## 📄 License

MIT (customize as needed for your organization).

Enjoy building with LangChain! 🎉


