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
    new_data_batch = [{
    "doc_id": "plus_14",
    "category": "System & Law",
    "title": "신라의 교육·관리 제도 (Silla Education & Official System)",
    "aliases": [
      "국학",
      "독서삼품과",
      "신문왕"
    ],
    "rag_matching_keywords": {
      "quotes": [
        "국학 설치",
        "유교 경전 교육",
        "독서삼품과",
        "임신서기석"
      ],
      "related_terms": [
        "신문왕",
        "원성왕",
        "인재 양성",
        "유학 보급"
      ]
    },
    "content": {
      "definition": "신라의 교육 기관 및 관리 선발 제도.",
      "achievements": [
        "국학(682, 신문왕): 유교 경전을 가르치는 국립 교육 기관 설치 -> 인재 양성 및 유학 보급",
        "독서삼품과(788, 원성왕): 유교 경전 독서 능력에 따라 관리를 상·중·하 3등급으로 나누어 선발 -> 골품제의 한계로 실효성 낮음",
        "임신서기석: 신라 젊은이들이 유교 경전 공부와 나라에 충성할 것을 맹세한 비석 -> 유학 보급의 증거"
      ],
      "historical_context": [
        "신라 중·하대 유학 발달의 증거로 최치원 등 6두품 유학자 성장"
      ]
    },
    "search_text": "신라 국학 신문왕 유교 경전 교육 독서삼품과 원성왕 관리 선발 인재 양성 임신서기석 유학 보급 6두품 최치원"
  }]
    save_to_corpus(new_data_batch)