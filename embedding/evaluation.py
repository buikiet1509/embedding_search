import json
 
from config import TESTSET_PATH, TOP_K
from index import get_database
 
# --- Bước 1-2: đọc testset và chia nhóm ---
with open(TESTSET_PATH, "r", encoding="utf-8") as file:
    testset = json.load(file)
 
answerable_items = [item for item in testset if item["answerable"]]
unanswerable_items = [item for item in testset if not item["answerable"]]
 
print(f"Tổng: {len(testset)} | Có đáp án: {len(answerable_items)} | Không đáp án: {len(unanswerable_items)}")
 
# --- Bước 3: lấy database (gọi 1 lần, ngoài vòng lặp) ---
database = get_database()