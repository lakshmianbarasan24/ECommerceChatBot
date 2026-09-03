# E-Commerce RAG Chatbot API & Evaluation Suite

A Retrieval-Augmented Generation (RAG) Chatbot API and evaluation framework built with **Python**, **FastAPI**, **LangChain**, **FAISS**, and **Google Gemini AI**. The application indexes an e-commerce knowledge base, provides context-constrained API responses, and includes a full evaluation suite with dataset quality auditing and RAG performance scoring.

---

## 🌟 Architecture & Walkthrough

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
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Evaluation & Quality Assurance Suite                                            │
│  - golden_dataset.json: Ground-truth Q&A test cases                              │
│  - dataset_quality_checker.py: Audit dataset duplicates, blanks & score (>=95%)  │
│  - evaluate_rag.py: Measure Faithfulness & Answer Correctness RAG metrics        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Core Repository Files

- **`knowledge.txt`**: E-commerce customer support documentation (Return window, shipping fees, warranty terms, support contact).
- **`rag_pipeline.py`**: Reads knowledge base, splits text using `RecursiveCharacterTextSplitter` (`CHUNK_SIZE`, `CHUNK_OVERLAP` from `.env`), generates vector embeddings, creates/persists FAISS `IndexFlatL2` index in `data/faiss/`, and performs similarity searches.
- **`main.py`**: FastAPI application exposing `GET /` and `POST /ask`. Enforces strict context prompting for `gemini-1.5-flash` to prevent hallucinations.
- **`golden_dataset.json`**: Ground-truth benchmark dataset formatted with `question` and `expected_answer` pairs (in-scope policies and out-of-scope queries).
- **`dataset_quality_checker.py`**: Audits dataset quality (exact & near duplicates, missing fields, minimum length constraints) and computes a percentage quality score (requires $\ge 95\%$ for Golden Dataset status).
- **`evaluate_rag.py`**: Evaluates RAG metrics (**Faithfulness** and **Answer Correctness**) across all questions in `golden_dataset.json` and outputs `evaluation_results.json`.
- **`test_api.py`**: FastAPI `TestClient` integration test suite.

---

## 📋 Prerequisites

- **Python 3.9+** (`python --version`)
- **pip** package manager
- **Google Gemini API Key** (Obtain from [Google AI Studio](https://aistudio.google.com/))

---

## ⚙️ Environment Configuration

Create or update your `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

*An [.env.example](file:///d:/Lakshmi/Agentic-AI/E-CommerceChatBot/.env.example) template is provided in the root directory.*

---

## 🚀 Setup & Execution Guide

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build FAISS Vector Index & Standalone Test

```bash
python rag_pipeline.py
```
*Generated files: `data/faiss/index.faiss` and `data/faiss/chunks.txt`.*

### 3. Verify Golden Dataset Quality (Quality Auditor)

Audit `golden_dataset.json` for duplicates, blanks, and quality score ($\ge 95\%$ required):

```bash
python dataset_quality_checker.py golden_dataset.json
```

### 4. Evaluate RAG Metrics (Faithfulness & Correctness)

Pass dataset questions through the RAG pipeline and measure performance metrics:

```bash
python evaluate_rag.py
```
*Generated report: `evaluation_results.json`.*

### 5. Run API Server

```bash
uvicorn main:app --reload --port 8000
```
- **API Base URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 6. Run API Test Suite

```bash
python test_api.py
```

---

## 📡 API Reference & Examples

### `POST /ask`

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

### cURL Command

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"How long do I have to return a product?\"}"
```
