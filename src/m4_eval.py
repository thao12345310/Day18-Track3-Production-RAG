"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation on provided QA pairs.

    Uses ragas 0.4.3 API with EvaluationDataset + SingleTurnSample.
    Requires OPENAI_API_KEY in environment for LLM-based metrics.

    Args:
        questions: List of user questions.
        answers: List of generated answers.
        contexts: List of retrieved context lists (one per question).
        ground_truths: List of reference/ground-truth answers.

    Returns:
        Dict with aggregate scores for 4 metrics + per_question EvalResult list.
    """
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # --- Build evaluation dataset ---
    samples = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=ctx,
                reference=gt,
            )
        )
    dataset = EvaluationDataset(samples=samples)

    # --- Configure LLM & embeddings for ragas ---
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    # --- Define metrics ---
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    # --- Run evaluation ---
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    # --- Extract results ---
    df = result.to_pandas()

    # Map ragas column names to our standard metric names
    col_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "llm_context_precision_with_reference": "context_precision",
        "context_recall": "context_recall",
        "llm_context_recall": "context_recall",
    }

    # Detect actual column names present in the dataframe
    metric_cols = {}
    for ragas_col, our_col in col_map.items():
        if ragas_col in df.columns:
            metric_cols[our_col] = ragas_col

    # Build per-question results
    per_question = []
    for idx, row in df.iterrows():
        f_score = float(row.get(metric_cols.get("faithfulness", ""), 0.0) or 0.0)
        ar_score = float(row.get(metric_cols.get("answer_relevancy", ""), 0.0) or 0.0)
        cp_score = float(row.get(metric_cols.get("context_precision", ""), 0.0) or 0.0)
        cr_score = float(row.get(metric_cols.get("context_recall", ""), 0.0) or 0.0)

        per_question.append(
            EvalResult(
                question=questions[idx],
                answer=answers[idx],
                contexts=contexts[idx],
                ground_truth=ground_truths[idx],
                faithfulness=f_score,
                answer_relevancy=ar_score,
                context_precision=cp_score,
                context_recall=cr_score,
            )
        )

    # Compute aggregate scores
    agg = {
        "faithfulness": mean([r.faithfulness for r in per_question]) if per_question else 0.0,
        "answer_relevancy": mean([r.answer_relevancy for r in per_question]) if per_question else 0.0,
        "context_precision": mean([r.context_precision for r in per_question]) if per_question else 0.0,
        "context_recall": mean([r.context_recall for r in per_question]) if per_question else 0.0,
        "per_question": per_question,
    }
    return agg


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    For each EvalResult, computes avg score across 4 metrics, sorts ascending,
    takes the bottom_n, then maps the worst metric to a diagnosis + suggested fix.

    Args:
        eval_results: List of per-question EvalResult objects.
        bottom_n: Number of worst-performing questions to analyze.

    Returns:
        List of dicts with question, worst_metric, score, diagnosis, suggested_fix.
    """
    if not eval_results:
        return []

    # --- Diagnostic Tree mapping ---
    diagnostic_tree = {
        "faithfulness": {
            "threshold": 0.85,
            "diagnosis": "LLM hallucinating",
            "suggested_fix": "Tighten prompt, lower temperature",
        },
        "context_recall": {
            "threshold": 0.75,
            "diagnosis": "Missing relevant chunks",
            "suggested_fix": "Improve chunking or add BM25",
        },
        "context_precision": {
            "threshold": 0.75,
            "diagnosis": "Too many irrelevant chunks",
            "suggested_fix": "Add reranking or metadata filter",
        },
        "answer_relevancy": {
            "threshold": 0.80,
            "diagnosis": "Answer doesn't match question",
            "suggested_fix": "Improve prompt template",
        },
    }

    # 1. Compute avg score for each result
    scored = []
    for r in eval_results:
        avg_score = mean([r.faithfulness, r.answer_relevancy,
                          r.context_precision, r.context_recall])
        scored.append((avg_score, r))

    # 2. Sort ascending by avg_score → take bottom_n
    scored.sort(key=lambda x: x[0])
    bottom = scored[:bottom_n]

    # 3. For each failed question, find worst metric and map to diagnosis
    failures = []
    for avg_score, r in bottom:
        metric_scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }

        # Find the metric with the lowest score
        worst_metric = min(metric_scores, key=metric_scores.get)
        worst_score = metric_scores[worst_metric]

        # Map to diagnosis from diagnostic tree
        diag = diagnostic_tree[worst_metric]

        failures.append({
            "question": r.question,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "threshold": diag["threshold"],
            "diagnosis": diag["diagnosis"],
            "suggested_fix": diag["suggested_fix"],
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
