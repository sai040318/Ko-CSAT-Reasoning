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
    new_data_batch = [
        {
            "doc_id": "figure_modern_leesarngseol",
            "category": "Modern Period",
            "title": "이상설 (Lee Sang-seol)",
            "aliases": ["헤이그 특사", "서전서숙"],
            "rag_matching_keywords": {
                "quotes": ["을사늑약의 부당성을 알리기 위해", "헤이그 만국 평화 회의"],
                "related_terms": ["대한광복군 정부 정통령", "성명회", "권업회"]
            },
            "content": {
                "definition": "구한말~일제 강점기의 독립운동가. 만주와 연해주를 무대로 활동함.",
                "features": [
                    "교육 구국: 북간도 용정에 민족 교육 기관인 '서전서숙' 설립",
                    "헤이그 특사(1907): 고종의 밀명을 받고 이준, 이위종과 함께 네덜란드 헤이그 만국 평화 회의에 파견됨",
                    "연해주 활동: 성명회/권업회 조직, 대한광복군 정부(1914) 수립하여 정통령에 선임됨"
                ]
            },
            "search_text": "이상설 서전서숙 헤이그 특사 이준 이위종 대한광복군 정부 연해주 북간도 용정 성명회 권업회"
        },
        {
            "doc_id": "system_ancient_council",
            "category": "Ancient History",
            "title": "삼국의 귀족 회의 (Aristocratic Councils)",
            "aliases": ["화백회의", "정사암회의", "제가회의"],
            "rag_matching_keywords": {
                "quotes": ["재상을 뽑을 때 이름 위에 도장을 찍어 정사암에 둠", "만장일치제", "국가의 중대사 결정"],
                "related_terms": ["상대등", "대대로", "상좌평"]
            },
            "content": {
                "definition": "삼국 시대 귀족들이 국가 중대사를 결정하던 합의 기구.",
                "features": [
                    "신라 화백회의: 만장일치제, 상대등이 의장, 왕위 계승이나 폐위 결정 (왕권 견제)",
                    "백제 정사암 회의: 재상 선출 시 이름을 적어 봉한 뒤 정사암 바위 위에 둠 (천의 존중)",
                    "고구려 제가회의: 고위 관료(대가)들이 모여 국정 논의, 수상인 대대로 선출"
                ]
            },
            "search_text": "삼국 귀족 회의 신라 화백회의 만장일치 상대등 백제 정사암 회의 재상 선출 바위 고구려 제가회의 대대로"
        },
        {
            "doc_id": "system_goryeo_special",
            "category": "Social System",
            "title": "향·소·부곡 (Special Administrative Districts)",
            "aliases": ["특수 행정 구역", "향소부곡"],
            "rag_matching_keywords": {
                "quotes": ["천하고 힘든 일을 맡게 했다", "고려 태조 때의 명령을 거역한 사람", "전정과 호구가 현의 규모가 되지 못하는 곳"],
                "related_terms": ["거주 이전의 자유 제한", "세금 부담 과다", "과거 응시 제한", "공주 명학소의 난(망이 망소이)"]
            },
            "content": {
                "definition": "고려 시대의 특수 행정 구역. 양인이지만 일반 군현민보다 차별받는 계층이 거주함.",
                "features": [
                    "향/부곡: 주로 농업에 종사하며 국가 곡물 생산 담당",
                    "소(所): 수공업이나 광업 등 특정 물품(먹, 종이, 금은 등) 생산 담당",
                    "차별: 세금 부담이 무겁고, 거주 이전의 자유가 없으며, 국자감 입학 및 과거 응시가 금지됨",
                    "소멸: 조선 전기부터 점차 일반 군현으로 승격되거나 소멸됨 (속현 면리제 정착)"
                ]
            },
            "search_text": "향 소 부곡 특수 행정 구역 차별 대우 망이 망소이의 난 공주 명학소 고려 태조 역 거주 이전 금지 과거 응시 제한"
        }
    ]

    save_to_corpus(new_data_batch)