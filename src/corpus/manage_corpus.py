import json
import os

# 저장할 파일 경로 설정
CORPUS_DIR = "src/corpus"
CORPUS_FILE = "corpus.json"
FILE_PATH = os.path.join(CORPUS_DIR, CORPUS_FILE)

def save_to_corpus(new_documents):
    """
    새로운 문서 리스트를 받아 corpus.json 파일에 누적 저장하는 함수
    """
    # 1. 기존 데이터 로드 (파일이 존재할 경우)
    existing_data = []
    if os.path.exists(FILE_PATH):
        try:
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                print(f"📖 기존 데이터 {len(existing_data)}개를 로드했습니다.")
        except json.JSONDecodeError:
            print("⚠️ 기존 파일이 손상되었거나 비어있습니다. 새로 시작합니다.")
            existing_data = []

    # 3. 중복 방지 (선택 사항: doc_id가 같은 게 있으면 덮어쓰거나 건너뛰기)
    # 여기서는 간단하게 기존 ID 목록을 뽑아서 중복 체크를 합니다.
    existing_ids = {doc["doc_id"] for doc in existing_data}
    
    added_count = 0
    for doc in new_documents:
        if doc["doc_id"] not in existing_ids:
            existing_data.append(doc)
            existing_ids.add(doc["doc_id"])
            added_count += 1
        else:
            print(f"중복된 doc_id 건너뜀: {doc['doc_id']}")

    # 4. 파일에 다시 저장
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f" {added_count}개의 새 문서가 저장되었습니다.")
    print(f" 총 문서 수: {len(existing_data)}개")
    print(f"💾저장 경로: {FILE_PATH}")

# ==========================================
# ▼ 아래 영역에 제가 만들어드리는 데이터를 붙여넣고 실행하면 됩니다. ▼
# ==========================================

if __name__ == "__main__":
    # 예시: 새로 추가할 문서 데이터
    new_data_batch = [
        {
            "doc_id": "doc-001",
            "title": "한국 역사 개요",
            "content": "한국의 역사는 고조선부터 시작하여 삼국시대, 고려, 조선, 현대에 이르기까지 다양합니다.",
            "metadata": {
                "source": "history_book_1",
                "date_added": "2024-06-01"
            }
        },
        {
            "doc_id": "doc-002",
            "title": "조선 시대의 문화",
            "content": "조선 시대는 유교 사상이 지배적이었으며, 한글 창제와 같은 중요한 문화적 발전이 있었습니다.",
            "metadata": {
                "source": "history_book_2",
                "date_added": "2024-06-01"
            }
        }
    ]
       # 저장 함수 실행
    save_to_corpus(new_data_batch)