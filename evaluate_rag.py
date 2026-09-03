import os
import json
import re
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from main import generate_fallback_answer, STOP_WORDS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RAGASEvaluator:
    def __init__(self, dataset_path: str = "golden_dataset.json"):
        self.dataset_path = dataset_path
        self.pipeline = RAGPipeline()
        self.pipeline.load_or_build()

        api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = None
        if api_key and api_key != "your_api_key":
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.0
                )
            except Exception as e:
                print(f"LLM initialization warning: {e}")

    def load_dataset(self) -> list[dict]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def calculate_faithfulness_heuristic(self, answer: str, context: list[str]) -> float:
        """Calculates faithfulness score: ratio of answer content words found in context."""
        if "I don't have enough information" in answer:
            return 1.0
        
        combined_context = " ".join(context).lower()
        ans_words = re.findall(r'\b[a-z]{3,}\b', answer.lower())
        content_words = [w for w in ans_words if w not in STOP_WORDS]
        
        if not content_words:
            return 1.0
            
        supported = [w for w in content_words if w in combined_context]
        return round(len(supported) / len(content_words), 2)

    def calculate_correctness_heuristic(self, generated_answer: str, expected_answer: str) -> float:
        """Calculates answer correctness score: key concept and token overlap with expected answer."""
        gen_clean = generated_answer.strip().lower()
        exp_clean = expected_answer.strip().lower()

        if gen_clean == exp_clean:
            return 1.0

        if "i don't have enough information" in exp_clean:
            if "i don't have enough information" in gen_clean:
                return 1.0
            return 0.0

        if "i don't have enough information" in gen_clean:
            return 0.0

        gen_tokens = set(re.findall(r'\b\w+\b', gen_clean)) - STOP_WORDS
        exp_tokens = set(re.findall(r'\b\w+\b', exp_clean)) - STOP_WORDS

        if not exp_tokens:
            return 1.0 if not gen_tokens else 0.0

        intersection = gen_tokens.intersection(exp_tokens)
        if not intersection:
            return 0.0

        # Ratio of key expected tokens covered in generated answer
        overlap_score = len(intersection) / len(exp_tokens)
        return round(min(1.0, overlap_score), 2)

    def evaluate_item_with_llm(self, question: str, answer: str, context: list[str], expected_answer: str) -> tuple[float, float]:
        """Evaluates Faithfulness and Correctness using Gemini LLM when available."""
        context_str = "\n".join(context)
        
        # 1. Faithfulness Prompt
        faith_prompt = f"""Rate the Faithfulness of the Generated Answer relative to the Context on a scale from 0.0 to 1.0.
Faithfulness measures if all statements in the answer are strictly supported by the Context.
Output ONLY a float number between 0.0 and 1.0.

Context:
{context_str}

Generated Answer: {answer}
Score:"""

        # 2. Correctness Prompt
        corr_prompt = f"""Rate the Answer Correctness of the Generated Answer compared to the Expected Answer on a scale from 0.0 to 1.0.
Output ONLY a float number between 0.0 and 1.0.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {answer}
Score:"""

        try:
            faith_res = self.llm.invoke(faith_prompt).content.strip()
            corr_res = self.llm.invoke(corr_prompt).content.strip()
            
            faith_score = float(re.findall(r"0\.\d+|1\.0|0|1", faith_res)[0])
            corr_score = float(re.findall(r"0\.\d+|1\.0|0|1", corr_res)[0])
            return faith_score, corr_score
        except Exception as e:
            # Fallback to heuristic
            return (
                self.calculate_faithfulness_heuristic(answer, context),
                self.calculate_correctness_heuristic(answer, expected_answer)
            )

    def run_evaluation(self):
        dataset = self.load_dataset()
        results = []
        
        total_faithfulness = 0.0
        total_correctness = 0.0

        print("\n" + "=" * 80)
        print("                  RAGAS METRICS EVALUATION REPORT                  ")
        print("=" * 80)
        print(f"{'#':<3} | {'Question':<35} | {'Faithfulness':<12} | {'Correctness':<11} | {'Status':<6}")
        print("-" * 80)

        for i, item in enumerate(dataset, 1):
            q = item["question"]
            exp_ans = item["expected_answer"]

            # Retrieve context chunks
            retrieved_chunks = self.pipeline.search(q, top_k=3)

            # Generate answer
            context_str = "\n".join(retrieved_chunks)
            if self.llm is not None:
                prompt = f"Answer strictly based on context:\nContext:\n{context_str}\nQuestion: {q}"
                try:
                    gen_ans = self.llm.invoke(prompt).content.strip()
                except Exception:
                    gen_ans = generate_fallback_answer(q, retrieved_chunks)
            else:
                gen_ans = generate_fallback_answer(q, retrieved_chunks)

            # Evaluate Metrics
            if self.llm is not None:
                faith, corr = self.evaluate_item_with_llm(q, gen_ans, retrieved_chunks, exp_ans)
            else:
                faith = self.calculate_faithfulness_heuristic(gen_ans, retrieved_chunks)
                corr = self.calculate_correctness_heuristic(gen_ans, exp_ans)

            total_faithfulness += faith
            total_correctness += corr

            status = "PASS" if faith >= 0.7 and corr >= 0.7 else "CHECK"

            results.append({
                "question": q,
                "expected_answer": exp_ans,
                "generated_answer": gen_ans,
                "contexts": retrieved_chunks,
                "faithfulness": faith,
                "correctness": corr,
                "status": status
            })

            q_disp = q[:32] + "..." if len(q) > 35 else q
            print(f"{i:<3} | {q_disp:<35} | {faith:<12.2f} | {corr:<11.2f} | {status:<6}")

        avg_faithfulness = round(total_faithfulness / len(dataset), 2)
        avg_correctness = round(total_correctness / len(dataset), 2)
        overall_score = round((avg_faithfulness + avg_correctness) / 2, 2)

        print("=" * 80)
        print(f"AVERAGE FAITHFULNESS SCORE : {avg_faithfulness:.2f} / 1.00")
        print(f"AVERAGE CORRECTNESS SCORE  : {avg_correctness:.2f} / 1.00")
        print(f"OVERALL RAG PERFORMANCE    : {overall_score * 100:.1f}%")
        print("=" * 80 + "\n")

        # Save evaluation summary
        output_file = "evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": {
                    "avg_faithfulness": avg_faithfulness,
                    "avg_correctness": avg_correctness,
                    "overall_score": overall_score
                },
                "details": results
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed evaluation results saved to '{output_file}'.\n")

if __name__ == "__main__":
    evaluator = RAGASEvaluator()
    evaluator.run_evaluation()
