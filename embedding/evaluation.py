import json
import unicodedata
from collections import defaultdict

from config import SCORE_THRESHOLD, TESTSET_PATH, TOP_K
from models import get_retriever

K_VALUES = [1, 3, 5, 10]
THRESHOLDS = [round(0.30 + 0.02 * i, 2) for i in range(16)]   # 0.30 .. 0.60


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def contain_evidence(text: str, evidence: list) -> bool:
    text = nfc(text)
    return any(nfc(e) in text for e in evidence)

def first_hit_rank(docs: list, evidence: list):
    for rank, doc in enumerate(docs, start=1):
        if contain_evidence(doc.page_content, evidence):
            return rank
    return None


def run_retrieval(testset: list, search) -> list:
    """Retrieve max(K_VALUES) chunks per query so every Hit@k can be computed from one search."""
    runs = []
    for item in testset:
        docs, top_score = search(item["query"], max(K_VALUES))
        runs.append({
            "item": item,
            "result": docs,
            "rank": first_hit_rank(docs, item["evidence"]) if item["answerable"] else None,
            "top_score": top_score,
        })
    return runs


def report_retrieval(runs: list, name: str) -> None:
    answerable = [run for run in runs if run["item"]["answerable"]]

    print(f"\n##### {name} #####")
    print(f"\n=== Retrieval ({len(answerable)} answerable queries) ===")
    for k in K_VALUES:
        hit = sum(1 for run in answerable if run["rank"] and run["rank"] <= k) / len(answerable)
        print(f"  Hit@{k:<3} {hit:.3f}")
    mrr = sum(1 / run["rank"] for run in answerable if run["rank"] and run["rank"] <= TOP_K) / len(answerable)
    print(f"  MRR@{TOP_K}  {mrr:.3f}")

    # --- per question type, at the k the chatbot actually uses ---
    by_type = defaultdict(list)
    for run in answerable:
        by_type[run["item"]["type"]].append(run)
    print(f"\n=== Hit@{TOP_K} by type ===")
    for qtype, group in sorted(by_type.items()):
        hits = sum(1 for run in group if run["rank"] and run["rank"] <= TOP_K)
        print(f"  {qtype:<14} {hits}/{len(group)}")

    misses = [run for run in answerable if not run["rank"] or run["rank"] > TOP_K]
    print(f"\n=== Missed queries at k={TOP_K} ({len(misses)}) ===")
    for run in misses:
        item = run["item"]
        found = f"found at rank {run['rank']}" if run["rank"] else f"not in top {max(K_VALUES)}"
        print(f"  {item['id']:<7} | {item['type']:<13} | {found:<16} | {item['query']}")


def report_threshold(runs: list) -> None:
    """How well does a top-1 score cutoff separate answerable from unanswerable queries?"""
    answerable = [run for run in runs if run["item"]["answerable"]]
    unanswerable = [run for run in runs if not run["item"]["answerable"]]

    print("\n=== Top-1 score distribution ===")
    for name, group in (("answerable", answerable), ("unanswerable", unanswerable)):
        scores = [run["top_score"] for run in group]
        print(f"  {name:<13} min {min(scores):.3f}   max {max(scores):.3f}")

    print("\n=== Unanswerable top-1 scores by type ===")
    for run in sorted(unanswerable, key=lambda r: -r["top_score"]):
        item = run["item"]
        print(f"  {item['id']:<7} | {item['type']:<11} | {run['top_score']:.3f} | {item['query']}")

    # correct refusal  = unanswerable query scored below the threshold
    # wrong refusal    = answerable query scored below the threshold
    print("\n=== Threshold sweep ===")
    print(f"  {'thr':<6} {'correct refusals':<18} {'wrong refusals':<15}")
    for thr in THRESHOLDS:
        correct = sum(1 for run in unanswerable if run["top_score"] < thr)
        wrong = sum(1 for run in answerable if run["top_score"] < thr)
        marker = "  <- current" if abs(thr - SCORE_THRESHOLD) < 1e-9 else ""
        print(f"  {thr:<6.2f} {correct}/{len(unanswerable):<16} {wrong}/{len(answerable):<13}{marker}")


def main():
    with open(TESTSET_PATH, "r", encoding="utf-8") as file:
        testset = json.load(file)

    retriever = get_retriever()
    for name, search in (("dense (FAISS)", retriever.dense_search),
                         ("hybrid (FAISS + BM25, RRF)", retriever.search)):
        runs = run_retrieval(testset, search)
        report_retrieval(runs, name)
    report_threshold(runs)   # top_score is the dense top-1 in both modes


if __name__ == "__main__":
    main()
