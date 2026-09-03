import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI(
    title="E-Commerce RAG Chatbot API",
    description="RAG-powered API using FAISS and Gemini models",
    version="1.0.0"
)

# Initialize RAG Pipeline
pipeline = RAGPipeline()
pipeline.load_or_build()

# Initialize Gemini Chat Model
api_key = os.getenv("GOOGLE_API_KEY")
llm = None
if api_key and api_key != "your_api_key":
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini LLM ({e}).")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: list[str]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the E-Commerce RAG Chatbot API!",
        "endpoints": {
            "POST /ask": "Submit customer question to receive RAG-generated answer"
        }
    }

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "where", "when", "why", "how", "who", "which", "whose",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "from", "to", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further", "then", "once",
    "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must"
}

def generate_fallback_answer(question: str, context_chunks: list[str]) -> str:
    """Fallback answer generator when live Gemini API key is unavailable."""
    combined_context = "\n".join(context_chunks).lower()
    
    # Clean question words and remove common stopwords
    q_words = re.findall(r'\b[a-z]{3,}\b', question.lower())
    content_words = [w for w in q_words if w not in STOP_WORDS]
    
    if not content_words:
        return "I don't have enough information to answer that question."

    # Count how many meaningful content words match in context
    matched_words = [w for w in content_words if w in combined_context]
    
    # Require at least 50% content word overlap for fallback retrieval
    if len(matched_words) < max(1, len(content_words) // 2):
        return "I don't have enough information to answer that question."

    matching_sentences = []
    for chunk in context_chunks:
        for line in chunk.split("\n"):
            line_clean = line.strip("- ").strip()
            if any(w in line_clean.lower() for w in matched_words):
                if line_clean and line_clean not in matching_sentences:
                    matching_sentences.append(line_clean)
                    
    if matching_sentences:
        return " ".join(matching_sentences)

    return "I don't have enough information to answer that question."

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: Retrieve top 3 context chunks via FAISS vector search
    retrieved_chunks = pipeline.search(question, top_k=3)
    context_text = "\n\n".join(retrieved_chunks)

    # Step 2: Formulate Gemini prompt with strict context constraint
    prompt = f"""You are an e-commerce customer support assistant.
Answer the customer's question based strictly ONLY on the following retrieved context.
If the answer is not available in the context, respond EXACTLY with:
"I don't have enough information to answer that question."

Do not use any external knowledge or make assumptions outside the provided context.

Retrieved Context:
{context_text}

Customer Question: {question}
Answer:"""

    answer = ""
    # Step 3: Generate answer via Gemini LLM or fallback
    if llm is not None:
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            print(f"Gemini API invocation failed ({e}). Using offline fallback response generator.")
            answer = generate_fallback_answer(question, retrieved_chunks)
    else:
        answer = generate_fallback_answer(question, retrieved_chunks)

    return AnswerResponse(
        question=question,
        answer=answer,
        retrieved_context=retrieved_chunks
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
