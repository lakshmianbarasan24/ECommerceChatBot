# E-Commerce RAG Chatbot API

A Retrieval-Augmented Generation (RAG) Chatbot API built with **Python**, **FastAPI**, **LangChain**, **FAISS**, and **Google Gemini AI**. The application processes an e-commerce knowledge base, indexes policy documents using vector embeddings, and provides accurate context-constrained answers via REST API endpoints.

---

## 🌟 Architecture & Walkthrough

The application follows a standard RAG workflow:

```text
┌─────────────────┐       ┌──────────────────────────┐       ┌──────────────────────┐
│  knowledge.txt  │ ────► │ RecursiveTextSplitter    │ ────► │  Gemini Embeddings   │
└─────────────────┘       └──────────────────────────┘       └──────────┬───────────┘
                                                                        │
                                                                        ▼
┌─────────────────┐       ┌──────────────────────────┐       ┌──────────────────────┐
│  FastAPI /ask   │ ◄───► │  FAISS Vector Search     │ ◄───► │  IndexFlatL2 Vector  │
│  Gemini Flash   │       │  Top 3 Chunks Retrieval  │       │  Store Persistence   │
└─────────────────┘       └──────────────────────────┘       └──────────────────────┘
```

### Core Components

- **`knowledge.txt`**: E-commerce policy document covering Return & Refund policies, Shipping & Delivery options, Order Tracking & Cancellations, Warranty & Repairs, Payment Methods, and Support channels.
- **`rag_pipeline.py`**:
  - Loads and splits `knowledge.txt` using `RecursiveCharacterTextSplitter` with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`.
  - Generates embeddings via Google Gemini (`models/embedding-001`) with automatic local fallback for offline development.
  - Builds and persists a FAISS `IndexFlatL2` vector index in `data/faiss/index.faiss` and `data/faiss/chunks.txt`.
  - Performs similarity searches to retrieve the top 3 relevant chunks for any question.
- **`main.py`**:
  - FastAPI application exposing REST endpoints.
  - Formulates a strict system prompt instructing `gemini-2.5-flash` to answer **only** using retrieved context.
  - Responds with `"I don't have enough information to answer that question."` when context is insufficient.
- **`test_api.py`**: Automated test suite utilizing FastAPI's `TestClient` for unit and integration testing.

---

## 📋 Prerequisites

Before running the application, ensure you have:

- **Python 3.9+** installed (`python --version`)
- **pip** package manager
- **Google Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/))

---

## ⚙️ Environment Configuration

Create or update the `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

*Note: An [.env.example](file:///d:/Lakshmi/Agentic-AI/E-CommerceChatBot/.env.example) template is provided in the repository.*

---

## 🚀 Setup & Execution Instructions

### 1. Install Dependencies

Install all required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Build the FAISS Vector Index

Run `rag_pipeline.py` to process `knowledge.txt`, generate vector embeddings, create the FAISS index, and execute a standalone test query:

```bash
python rag_pipeline.py
```

*Output files generated:*
- `data/faiss/index.faiss` (FAISS vector store)
- `data/faiss/chunks.txt` (JSON-encoded text chunks)

### 3. Start the FastAPI Application Server

Launch the web server using `uvicorn`:

```bash
uvicorn main:app --reload --port 8000
```

The server will start at `http://localhost:8000`.

### 4. Interactive API Documentation (Swagger UI)

Access the interactive API documentation by opening your browser at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 5. Execute Automated Test Suite

Run the integration tests to verify all endpoints and strict fallback logic:

```bash
python test_api.py
```

---

## 📡 API Endpoint Reference

### `GET /`
Health check and endpoint overview.

**Response:**
```json
{
  "status": "online",
  "message": "Welcome to the E-Commerce RAG Chatbot API!",
  "endpoints": {
    "POST /ask": "Submit customer question to receive RAG-generated answer"
  }
}
```

---

### `POST /ask`
Submit a question to retrieve relevant context and generate a RAG-backed response.

**Request Body:**
```json
{
  "question": "How long do I have to return a product?"
}
```

**Response (In-Scope Question):**
```json
{
  "question": "How long do I have to return a product?",
  "answer": "Customers have up to 30 days from the delivery date to return any item purchased from our online store.",
  "retrieved_context": [
    "1. RETURN & REFUND POLICY\n- Return Window: Customers have up to 30 days from the delivery date to return any item purchased from our online store.",
    "- Return Process: To initiate a return, log into your account, navigate to \"Orders\", select the order, and click \"Initiate Return\".",
    "- Return Shipping Fees: Standard return shipping is free for all orders above $50."
  ]
}
```

**Response (Out-of-Scope Question):**
```json
{
  "question": "What is the capital of France?",
  "answer": "I don't have enough information to answer that question.",
  "retrieved_context": [...]
}
```

---

## 🧪 Example cURL Request

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"How long do I have to return a product?\"}"
```
