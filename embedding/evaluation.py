import json
import unicodedata

from config import TESTSET_PATH, TOP_K
from index import get_database
 
# --- Steps 1-2: load the testset and split into groups ---
with open(TESTSET_PATH, "r", encoding="utf-8") as file:
    testset = json.load(file)
 
answerable_items = [item for item in testset if item["answerable"]]
unanswerable_items = [item for item in testset if not item["answerable"]]
 
runs = []
database = get_database()
for item in testset:
    result = database.similarity_search_with_score(item["query"], k=TOP_K)
    runs.append({
        "item" : item,
        "result": result
    })

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def contain_evidence(text: str, evidence: list) -> bool:
    text = nfc(text)
    return any(nfc(e) in text for e in evidence)

def first_hit_rank(results: list, evidence:list):
    for rank, (doc,score) in enumerate(results, start=1):
        if contain_evidence(doc.page_content, evidence):
            return rank
    return None

hits, rrs, misses = [],[],[]

for run in runs: 
    item = run["item"]
    if not item["answerable"]:
        continue
    
    rank = first_hit_rank(run["result"], item["evidence"])

    if rank is None:
        hits.append(0)
        rrs.append(0)
        misses.append(item)
    else:
        hits.append(1)
        rrs.append(1/rank)

hit_at_k = sum(hits) / len(hits)
mrr = sum(rrs) / len(rrs)

print(f"\n=== Retrieval ({len(hits)} answerable queries, k={TOP_K}) ===")
print(f"Hit@{TOP_K}: {hit_at_k:.3f}   MRR: {mrr:.3f}")

print(f"\n=== Missed queries ({len(misses)}) ===")
for item in misses:
    print(f"  {item['id']:<7} | {item['type']:<13} | {item['query']}")