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
        # Phase 7-2: 선사 시대 예술 및 암각화 (추가본)
        # Phase 7-4: 문화, 종교, 인물사 보강 (Final Patch)
    # Phase 7-5: 근대 언론, 민족 사학, 60~70년대 경제 (Final Finish)
    new_data_batch = [
        # 1. 근대 언론
        {
            "doc_id": "culture_modern_press",
            "category": "Modern Period",
            "title": "근대 신문과 언론 활동 (Modern Press)",
            "aliases": ["황성신문", "대한매일신보", "독립신문", "제국신문"],
            "rag_matching_keywords": {
                "quotes": ["시일야방성대곡", "장지연", "국채 보상 운동 확산", "베델 양기탁"],
                "related_terms": ["을사조약 비판", "신문지법 탄압", "순 한글 신문"]
            },
            "content": {
                "definition": "개항기 및 구한말 애국 계몽 운동을 이끈 주요 신문들.",
                "features": [
                    "황성신문(남궁억, 장지연): 국한문 혼용체, 을사조약 때 장지연의 '시일야방성대곡' 게재 (유림 층 구독)",
                    "대한매일신보(양기탁, 베델): 영국인 베델이 사장이라 일제의 검열을 피함 -> 의병 투쟁 호의적 보도, 국채 보상 운동 주도",
                    "제국신문: 순 한글 간행 (부녀자, 서민 층 구독)"
                ]
            },
            "search_text": "근대 신문 황성신문 장지연 시일야방성대곡 을사조약 비판 대한매일신보 베델 양기탁 국채 보상 운동"
        },
        # 2. 민족주의 사학 (정인보 등)
        {
            "doc_id": "figure_modern_historians",
            "category": "Modern Period",
            "title": "민족주의 사학자 (Nationalist Historians)",
            "aliases": ["정인보", "문일평", "안재홍"],
            "rag_matching_keywords": {
                "quotes": ["얼을 강조", "조선심", "5천년간 조선의 얼", "여유당전서 간행"],
                "related_terms": ["신채호 계승", "조선학 운동", "다산 서거 100주년"]
            },
            "content": {
                "definition": "1930년대 일제의 식민 사관에 맞서 우리 민족의 고유한 정신(얼, 심)을 강조한 역사학자들.",
                "features": [
                    "정인보: '조선의 얼' 강조, 광개토대왕릉비 연구, 양명학 연구",
                    "문일평: '조선심' 강조 (세종대왕 주목)",
                    "조선학 운동(1934): 정약용 서거 100주년을 맞아 '여유당전서' 간행, 실학 연구 심화"
                ]
            },
            "search_text": "민족주의 사학 정인보 얼 조선의 얼 문일평 조선심 안재홍 조선학 운동 여유당전서 정약용"
        },
        # 3. 만민공동회 (독립협회)
        {
            "doc_id": "org_modern_05", # 기존 org_modern_... 와 안 겹치게 ID 확인
            "category": "Modern Period",
            "title": "독립협회와 만민공동회 (Independence Club)",
            "aliases": ["관민공동회", "서재필", "헌의 6조"],
            "rag_matching_keywords": {
                "quotes": ["종로 네거리 토론회", "러시아 절영도 조차 저지", "자유 민권 운동", "의회 설립 운동"],
                "related_terms": ["중추원 개편", "입헌 군주제 지향", "고종의 해산 명령"]
            },
            "content": {
                "definition": "1896년 서재필 등이 창립한 자주 독립, 자유 민권, 자강 개혁 단체.",
                "features": [
                    "만민공동회: 종로에서 열린 최초의 근대적 민중 집회 (러시아 간섭 배격, 이권 수호)",
                    "관민공동회: 정부 관리와 백성이 함께 토론 -> '헌의 6조' 결의 (고종 수용)",
                    "해산: 공화정을 추진한다는 모함을 받아 황국협회(보부상)의 습격으로 강제 해산됨"
                ]
            },
            "search_text": "독립협회 만민공동회 종로 네거리 토론회 관민공동회 헌의 6조 서재필 자주 독립 자유 민권 러시아 이권 침탈 저지"
        },
        # 4. 현대 경제 (경제개발 5개년)
        {
            "doc_id": "economy_contemporary_01",
            "category": "Contemporary Period",
            "title": "경제 개발 5개년 계획 (Five-Year Economic Plans)",
            "aliases": ["수출 주도 산업화", "한강의 기적"],
            "rag_matching_keywords": {
                "quotes": ["1962년 시작", "경부 고속도로 개통", "포항 제철소", "새마을 운동"],
                "related_terms": ["박정희 정부", "전태일 분신", "저임금 저곡가"]
            },
            "content": {
                "definition": "1962년부터 박정희 정부 주도로 추진된 정부 주도형 경제 성장 정책.",
                "features": [
                    "1,2차(1962~1971): 경공업 중심 수출 증대, 사회 간접 자본(SOC) 확충 (경부고속도로)",
                    "3,4차(1972~1981): 중화학 공업 육성 (포항제철, 현대조선), 석유 파동 극복",
                    "성과와 그늘: 고도 성장(한강의 기적) 달성, 빈부 격차 심화, 노동 문제 발생(전태일)"
                ]
            },
            "search_text": "경제 개발 5개년 계획 1962년 박정희 수출 주도 경공업 중화학 공업 경부 고속도로 한강의 기적"
        }
    ]
# save_to_corpus(last_mile_batch)
# save_to_corpus(culture_batch)
# save_to_corpus(pre_art_batch)
       # 저장 함수 실행
    save_to_corpus(new_data_batch)