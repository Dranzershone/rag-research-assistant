"""
evaluation.py — RAG Evaluation using RAGAS + TruLens
Metrics: faithfulness, answer_relevancy, context_precision, context_recall
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str] = None,
) -> Dict[str, float]:
    """
    Evaluate RAG pipeline using RAGAS metrics.

    Args:
        questions:     list of user questions
        answers:       list of generated answers
        contexts:      list of retrieved context lists (per question)
        ground_truths: optional list of reference answers

    Returns:
        dict of metric scores (0.0 – 1.0, higher is better)
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset

    data: Dict[str, List] = {
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    }
    metrics = [faithfulness, answer_relevancy, context_precision]

    if ground_truths:
        data["ground_truth"] = ground_truths
        metrics.append(context_recall)

    dataset = Dataset.from_dict(data)
    logger.info(f"Running RAGAS on {len(questions)} samples...")
    result = evaluate(dataset, metrics=metrics)

    scores = {k: round(float(v), 4) for k, v in result.items()}
    logger.info(f"RAGAS scores: {scores}")
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SINGLE-QUERY EVAL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single(
    question: str,
    answer: str,
    source_docs: List[Document],
    ground_truth: str = None,
) -> Dict[str, float]:
    """Convenience wrapper to evaluate one Q&A pair."""
    contexts = [[doc.page_content for doc in source_docs]]
    return evaluate_with_ragas(
        questions=[question],
        answers=[answer],
        contexts=contexts,
        ground_truths=[ground_truth] if ground_truth else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BATCH EVAL FROM FILE
# ══════════════════════════════════════════════════════════════════════════════

def run_batch_evaluation(rag_chain, eval_data: List[Dict]) -> Dict[str, Any]:
    """
    Run batch evaluation given a list of dicts:
      [{"question": "...", "ground_truth": "..."}, ...]

    Runs the RAG chain on each question, collects results, evaluates with RAGAS.
    Saves results to evaluation/results.json.
    """
    import json, os
    from pathlib import Path

    questions, answers, contexts, ground_truths = [], [], [], []

    logger.info(f"Running batch evaluation on {len(eval_data)} questions...")

    for item in eval_data:
        q = item["question"]
        result = rag_chain.invoke(q)
        questions.append(q)
        answers.append(result["answer"])
        contexts.append([d.page_content for d in result.get("source_documents", [])])
        if "ground_truth" in item:
            ground_truths.append(item["ground_truth"])

    scores = evaluate_with_ragas(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths if ground_truths else None,
    )

    report = {
        "num_samples": len(questions),
        "scores": scores,
        "details": [
            {"question": q, "answer": a, "scores": {}}
            for q, a in zip(questions, answers)
        ],
    }

    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/results.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved to evaluation/results.json")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY SCORES
# ══════════════════════════════════════════════════════════════════════════════

def print_scores(scores: Dict[str, float]) -> None:
    print("\n── RAGAS Evaluation Results ────────────────────────")
    for metric, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<25} {bar}  {score:.4f}")
    print("────────────────────────────────────────────────────\n")
