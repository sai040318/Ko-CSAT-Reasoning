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
            print(f"ℹ️ 중복된 doc_id 건너뜀: {doc['doc_id']}")

    # 4. 파일에 다시 저장
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"✅ {added_count}개의 새 문서가 저장되었습니다.")
    print(f"📊 총 문서 수: {len(existing_data)}개")
    print(f"💾 저장 경로: {FILE_PATH}")

# ==========================================
# ▼ 아래 영역에 제가 만들어드리는 데이터를 붙여넣고 실행하면 됩니다. ▼
# ==========================================

if __name__ == "__main__":
    
    # 예시 데이터 (테스트용)
    # Phase 1-1: 고구려 핵심 통치자 (ruler_goguryeo_01 ~ 07)
   # Phase 1-2: 백제 핵심 통치자 8인 (ruler_baekje_01 ~ 08)
    # Phase 1-3: 신라(상대) 핵심 통치자 8인 (ruler_silla_01 ~ 08)
    # Phase 1-4: 가야 핵심 통치자 2인 (ruler_gaya_01 ~ 02)
    # Phase 1-5: 통일 신라 핵심 통치자 6인 (ruler_silla_unified_01 ~ 06)
    new_data_batch = [
        # ================= [신라 중대: 전제 왕권 강화] =================
        {
            "doc_id": "ruler_silla_unified_01",
            "category": "Rulers",
            "title": "신라 태종 무열왕 (King Muyeol)",
            "aliases": ["김춘추", "최초의 진골 왕"],
            "rag_matching_keywords": {
                "quotes": ["진골 출신으로 처음 왕위에 오르다", "당나라에 청병하여 백제를 멸망시킴", "김유신과 결탁"],
                "related_terms": ["나당 연합", "갈문왕 폐지", "집사부 시중 강화"]
            },
            "content": {
                "definition": "신라 제29대 왕(김춘추). 진골 출신 최초의 왕으로 삼국 통일의 전쟁을 시작함.",
                "achievements": [
                    "삼국 통일 전쟁: 당나라와 연합(나당 동맹)하여 백제 멸망(660)",
                    "왕권 강화: 왕권을 위협하던 갈문왕 제도 폐지, 집사부(왕명 출납) 기능 강화 및 시중의 권한 강화 (상대등 견제)"
                ],
                "historical_context": [
                    "즉위 배경: 성골 남자가 끊기자 진골 출신으로 추대됨"
                ]
            },
            "search_text": "신라 태종 무열왕 김춘추 진골 최초 왕 나당 연합 백제 멸망 갈문왕 폐지 집사부 시중 강화 왕권 강화"
        },
        {
            "doc_id": "ruler_silla_unified_02",
            "category": "Rulers",
            "title": "신라 문무왕 (King Munmu)",
            "aliases": ["법민", "삼국 통일 완성", "해중릉"],
            "rag_matching_keywords": {
                "quotes": ["죽어서 동해의 용이 되어 나라를 지키겠다", "대왕암", "매소성 기벌포"],
                "related_terms": ["감은사", "나당 전쟁", "고구려 멸망", "외사정"]
            },
            "content": {
                "definition": "신라 제30대 왕. 고구려를 멸망시키고 당나라 세력을 축출하여 삼국 통일을 완성함.",
                "achievements": [
                    "삼국 통일 체제 완성(676): 고구려 멸망(668) -> 나당 전쟁 승리(매소성 전투, 기벌포 해전) -> 당나라 축출",
                    "체제 정비: 지방 감찰관인 외사정 파견"
                ],
                "historical_context": [
                    "대왕암(해중릉): 유언에 따라 화장하여 동해 바다에 장사 지냄 (왜구 격퇴 염원)",
                    "감은사: 부왕의 은혜에 감사한다는 뜻으로 짓기 시작 (신문왕 때 완공)"
                ]
            },
            "search_text": "신라 문무왕 삼국 통일 완성 고구려 멸망 나당 전쟁 매소성 전투 기벌포 해전 당나라 축출 동해 용 대왕암 감은사 외사정"
        },
        {
            "doc_id": "ruler_silla_unified_03",
            "category": "Rulers",
            "title": "신라 신문왕 (King Sinmun)",
            "aliases": ["전제 왕권 확립", "정명"],
            "rag_matching_keywords": {
                "quotes": ["김흠돌의 난을 진압", "만파식적", "녹읍을 폐지하고 관료전을 지급", "9주 5소경"],
                "related_terms": ["국학", "9서당 10정", "달구벌 천도 시도", "설총 화왕계"]
            },
            "content": {
                "definition": "신라 제31대 왕. 귀족 숙청과 제도 개혁을 통해 신라 전제 왕권의 최전성기를 이룸.",
                "achievements": [
                    "왕권 강화: 김흠돌(장인)의 모역 사건을 계기로 진골 귀족 대거 숙청",
                    "경제 개혁: 관료전 지급(수조권만 인정) 및 녹읍 폐지 (귀족의 경제/군사 기반 약화)",
                    "행정/군사: 전국 9주 5소경 정비, 9서당(중앙군, 민족 융합) 10정(지방군) 완비",
                    "교육: 국학 설립 (유학 교육)"
                ],
                "historical_context": [
                    "만파식적 설화: 불면 적병이 물러가는 피리, 왕권 안정을 상징",
                    "설총의 화왕계: 왕에게 바른 정치를 조언",
                    "달구벌(대구) 천도 시도: 귀족 반발로 실패"
                ]
            },
            "search_text": "신라 신문왕 전제 왕권 강화 김흠돌의 난 숙청 관료전 지급 녹읍 폐지 9주 5소경 9서당 10정 국학 설립 만파식적 설총 화왕계 달구벌 천도 시도"
        },
        {
            "doc_id": "ruler_silla_unified_04",
            "category": "Rulers",
            "title": "신라 경덕왕 (King Gyeongdeok)",
            "aliases": ["문화 전성기", "녹읍 부활"],
            "rag_matching_keywords": {
                "quotes": ["불국사와 석굴암을 조성", "녹읍이 다시 부활", "관직과 지명을 한자식으로 고침"],
                "related_terms": ["김대성", "성덕대왕신종", "한화 정책"]
            },
            "content": {
                "definition": "신라 제35대 왕. 통일 신라의 불교 예술이 절정에 달했으나, 귀족 세력의 반발로 왕권이 약화되기 시작함.",
                "achievements": [
                    "문화 융성: 불국사, 석굴암 창건 (김대성), 성덕대왕신종(에밀레종) 주조 시작",
                    "한화 정책: 전국의 지명과 관직명을 중국식(한자 2글자)으로 변경 (왕권 강화 시도)"
                ],
                "historical_context": [
                    "왕권 약화: 귀족들의 반발로 녹읍 부활(757) -> 귀족 권한 강화, 왕권 쇠퇴의 신호탄"
                ]
            },
            "search_text": "신라 경덕왕 불국사 석굴암 김대성 성덕대왕신종 에밀레종 녹읍 부활 왕권 약화 한화 정책 지명 변경 중대"
        },

        # ================= [신라 하대: 혼란기] =================
        {
            "doc_id": "ruler_silla_unified_05",
            "category": "Rulers",
            "title": "신라 원성왕 (King Wonseong)",
            "aliases": ["독서삼품과"],
            "rag_matching_keywords": {
                "quotes": ["유교 경전의 이해 수준에 따라 관리를 등용", "독서삼품과", "상·중·하품"],
                "related_terms": ["골품제", "진골 귀족 반발", "국학"]
            },
            "content": {
                "definition": "신라 제38대 왕. 하대 초기에 왕권 강화를 위해 능력 중심의 관리 등용을 시도함.",
                "achievements": [
                    "독서삼품과 실시(788): 국학 학생들의 유교 경전 독해 능력을 3등급(상,중,하)으로 평가하여 관리 채용"
                ],
                "historical_context": [
                    "실패: 진골 귀족들의 반발과 골품제의 한계로 제대로 시행되지 못함"
                ]
            },
            "search_text": "신라 원성왕 독서삼품과 유교 경전 관리 등용 능력 중심 골품제 한계 하대"
        },
        {
            "doc_id": "ruler_silla_unified_06",
            "category": "Rulers",
            "title": "신라 진성여왕 (Queen Jinseong)",
            "aliases": ["신라 하대 혼란", "3대목"],
            "rag_matching_keywords": {
                "quotes": ["최치원이 시무 10조를 올림", "원종과 애노의 난", "적고적(붉은 바지 도적)", "세금 독촉"],
                "related_terms": ["향가집 편찬 시도(각간 위홍,대구화상)", "각간 위홍", "후삼국 성립 배경"]
            },
            "content": {
                "definition": "신라 제51대 왕. 신라 하대 극심한 혼란기로, 각지에서 농민 반란이 일어나고 후삼국 분열의 조짐이 보임.",
                "achievements": [
                    "문화: 각간 위홍과 대구화상에게 향가집 편찬 시도 (현재 전해지지 않음)"
                ],
                "historical_context": [
                    "농민 반란: 재정 파탄으로 관리를 보내 세금을 독촉하자 상주에서 원종·애노의 난 발생",
                    "도적 창궐: 적고적(붉은 바지를 입은 도적) 등 난립",
                    "최치원: 당나라에서 귀국하여 개혁안 '시무 10조'를 건의했으나 진골 귀족의 반대로 수용 안 됨"
                ]
            },
            "search_text": "신라 진성여왕 하대 혼란 원종 애노의 난 농민 반란 적고적 최치원 시무 10조 개혁 실패 삼대목 향가집"
        }
    ]
    # 저장 함수 실행
    save_to_corpus(new_data_batch)