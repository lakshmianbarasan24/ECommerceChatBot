import json
import sys
import os
import re
from typing import Dict, Any, List, Tuple

class DatasetQualityChecker:
    def __init__(self, dataset_path: str = "golden_dataset.json"):
        self.dataset_path = dataset_path

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads and parses the dataset JSON file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at path: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Dataset JSON root must be a list/array of objects.")
        return data

    def normalize_text(self, text: str) -> str:
        """Normalizes text for duplicate and quality comparison."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text)

    def evaluate_quality(self) -> Tuple[float, List[str], Dict[str, Any]]:
        """
        Evaluates the dataset across quality criteria:
        1. Required keys presence ('question', 'expected_answer')
        2. Empty/blank field checks
        3. Exact and normalized duplicate questions check
        4. Minimum length & structural formatting checks
        """
        records = self.load_dataset()
        total_count = len(records)
        if total_count == 0:
            return 0.0, ["Dataset is completely empty!"], {"total": 0, "valid": 0}

        issues: List[str] = []
        passed_count = 0

        seen_normalized_questions = set()
        seen_exact_questions = set()

        for idx, item in enumerate(records, 1):
            item_issues = []

            # 1. Type check
            if not isinstance(item, dict):
                issues.append(f"Item #{idx}: Entry is not a valid JSON object.")
                continue

            # 2. Required fields check
            if "question" not in item:
                item_issues.append("Missing required field 'question'.")
            if "expected_answer" not in item:
                item_issues.append("Missing required field 'expected_answer'.")

            if item_issues:
                issues.extend([f"Item #{idx}: {issue}" for issue in item_issues])
                continue

            q = item["question"]
            ans = item["expected_answer"]

            # 3. Non-string type check
            if not isinstance(q, str):
                item_issues.append("'question' field must be a string.")
            if not isinstance(ans, str):
                item_issues.append("'expected_answer' field must be a string.")

            if item_issues:
                issues.extend([f"Item #{idx}: {issue}" for issue in item_issues])
                continue

            # 4. Empty / whitespace check
            q_stripped = q.strip()
            ans_stripped = ans.strip()

            if not q_stripped:
                item_issues.append("Question is empty or whitespace-only.")
            elif len(q_stripped) < 5:
                item_issues.append(f"Question is too short ({len(q_stripped)} chars, min 5 required).")

            if not ans_stripped:
                item_issues.append("Expected answer is empty or whitespace-only.")
            elif len(ans_stripped) < 3:
                item_issues.append(f"Expected answer is too short ({len(ans_stripped)} chars, min 3 required).")

            # 5. Duplicate Question check
            q_norm = self.normalize_text(q_stripped)
            if q_stripped in seen_exact_questions:
                item_issues.append(f"Exact duplicate question detected: '{q_stripped}'")
            elif q_norm in seen_normalized_questions:
                item_issues.append(f"Near-duplicate question detected: '{q_stripped}'")

            seen_exact_questions.add(q_stripped)
            seen_normalized_questions.add(q_norm)

            if item_issues:
                issues.extend([f"Item #{idx}: {issue}" for issue in item_issues])
            else:
                passed_count += 1

        quality_score = (passed_count / total_count) * 100.0
        stats = {
            "total_items": total_count,
            "passed_items": passed_count,
            "failed_items": total_count - passed_count,
            "quality_score": round(quality_score, 2),
            "is_golden": quality_score >= 95.0
        }

        return quality_score, issues, stats

    def print_report(self):
        """Prints a visual dataset quality report."""
        print("=" * 60)
        print("         DATASET QUALITY CHECKER & EVALUATOR         ")
        print("=" * 60)
        print(f"Target Dataset File: {self.dataset_path}\n")

        try:
            score, issues, stats = self.evaluate_quality()
        except Exception as e:
            print(f" ERROR: Failed to evaluate dataset: {e}")
            return

        print(f"Total Entries Evaluated : {stats['total_items']}")
        print(f"Valid / Passed Entries  : {stats['passed_items']}")
        print(f"Failed Entries          : {stats['failed_items']}")
        print("-" * 60)
        print(f"DATASET QUALITY SCORE    : {stats['quality_score']}%")
        print("-" * 60)

        if issues:
            print("\n⚠️ ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n No quality issues found. All records passed validation!")

        print("\n" + "=" * 60)
        if stats["is_golden"]:
            print(" RESULT: APPROVED - DATASET QUALIFIES AS 'GOLDEN DATASET' (Score >= 95%)")
        else:
            print(" RESULT: REJECTED - DATASET DOES NOT MEET GOLDEN STANDARD (Score < 95%)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "golden_dataset.json"
    checker = DatasetQualityChecker(dataset_file)
    checker.print_report()
