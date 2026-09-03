import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class FallbackEmbeddings:
    """Fallback embedding generator for offline testing or when GOOGLE_API_KEY is unset."""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def _embed_text(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            idx = abs(hash(word)) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


class RAGPipeline:
    def __init__(
        self,
        knowledge_path: str = "knowledge.txt",
        index_dir: str = "data/faiss",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model: str = "models/text-embedding-004"
    ):
        self.knowledge_path = knowledge_path
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.chunks_path = os.path.join(index_dir, "chunks.txt")
        
        env_chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        env_chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        
        self.chunk_size = chunk_size if chunk_size is not None else env_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else env_chunk_overlap
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key != "your_api_key":
            print(f"Initializing Gemini Embeddings ({embedding_model})...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=api_key
            )
        else:
            print("WARNING: Valid GOOGLE_API_KEY not found in .env. Using fallback embeddings model.")
            self.embeddings = FallbackEmbeddings(dimension=768)

        self.index = None
        self.chunks = []

    def load_knowledge(self) -> str:
        """Reads the knowledge.txt file."""
        if not os.path.exists(self.knowledge_path):
            raise FileNotFoundError(f"Knowledge file not found at {self.knowledge_path}")
        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            return f.read()

    def split_text(self, text: str) -> list[str]:
        """Splits text into chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)

    def build_index(self):
        """Reads knowledge file, splits text, computes embeddings, and creates FAISS IndexFlatL2 index."""
        print(f"Loading knowledge base from '{self.knowledge_path}'...")
        text = self.load_knowledge()
        
        print("Splitting text into chunks...")
        self.chunks = self.split_text(text)
        print(f"Generated {len(self.chunks)} text chunks.")

        print("Generating embeddings...")
        try:
            embeddings_list = self.embeddings.embed_documents(self.chunks)
        except Exception as e:
            print(f"Gemini embedding call failed ({e}). Falling back to local embeddings...")
            self.embeddings = FallbackEmbeddings(dimension=768)
            embeddings_list = self.embeddings.embed_documents(self.chunks)

        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        dimension = embeddings_matrix.shape[1]
        print(f"Creating FAISS IndexFlatL2 index with dimension {dimension}...")
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_matrix)

        self.save_index()
        print("Index build complete!")

    def save_index(self):
        """Saves FAISS index to data/faiss/index.faiss and chunks to data/faiss/chunks.txt."""
        os.makedirs(self.index_dir, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            print(f"Saved FAISS index to '{self.index_path}'.")

        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved chunks to '{self.chunks_path}'.")

    def load_index(self) -> bool:
        """Loads an existing FAISS index and chunk metadata if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            print(f"Loading existing FAISS index from '{self.index_path}'...")
            self.index = faiss.read_index(self.index_path)
            
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks.")
            return True
        return False

    def load_or_build(self):
        """Loads existing index if present, otherwise builds a new one."""
        if not self.load_index():
            print("No existing index found. Building index...")
            self.build_index()

    def search(self, question: str, top_k: int = 3) -> list[str]:
        """Searches top_k relevant text chunks for a given question."""
        if self.index is None or len(self.chunks) == 0:
            self.load_or_build()

        try:
            query_embedding = self.embeddings.embed_query(question)
        except Exception as e:
            print(f"Gemini query embedding call failed ({e}). Falling back to local embeddings...")
            self.embeddings = FallbackEmbeddings(dimension=self.index.d)
            query_embedding = self.embeddings.embed_query(question)

        query_matrix = np.array([query_embedding], dtype=np.float32)

        distances, indices = self.index.search(query_matrix, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results


if __name__ == "__main__":
    print("=== Testing RAGPipeline ===")
    pipeline = RAGPipeline()
    pipeline.build_index()

    test_question = "How long do I have to return a product?"
    print(f"\nSearching for query: '{test_question}'")
    top_chunks = pipeline.search(test_question, top_k=3)

    print("\n--- Top 3 Retrieved Chunks ---")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\n[Chunk {i}]:\n{chunk}")
