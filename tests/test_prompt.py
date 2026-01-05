"""
qwen3_2507_thinking_ver1.0.0.txt 프롬프트 통합 테스트

ollama_prompt.py의 파싱 기능 및 선택지 개수별 프롬프트 생성을 종합적으로 테스트합니다.
- 파싱 기능 테스트 (SYSTEM, USER, ASSISTANT 태그)
- 선택지 2, 3, 4, 5개일 때 프롬프트 생성 검증
- 실제 프롬프트 출력 표시
"""

import sys
from pathlib import Path
from typing import List, Dict

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt.ollama_prompt import OllamaPromptBuilder


# ============================================================================
# 파트 1: 파싱 기능 테스트
# ============================================================================


def test_parse_chat_template_basic():
    """기본 파싱 테스트: SYSTEM, USER, ASSISTANT 모두 포함"""
    print("\n" + "=" * 80)
    print("파싱 테스트 1: 기본 태그 파싱 (SYSTEM, USER, ASSISTANT)")
    print("=" * 80)

    template = """<SYSTEM>
시스템 메시지입니다.
</SYSTEM>

<USER>
사용자 메시지입니다.
</USER>

<ASSISTANT>
어시스턴트 메시지입니다.
</ASSISTANT>"""

    messages = OllamaPromptBuilder._parse_chat_template(template)

    print(f"파싱된 메시지 수: {len(messages)}")
    for msg in messages:
        print(f"  - role: {msg['role']}, content 길이: {len(msg['content'])}")

    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    print("✓ 기본 파싱 테스트 통과")


def test_parse_chat_template_without_closing_tag():
    """닫는 태그가 없는 경우 테스트"""
    print("\n" + "=" * 80)
    print("파싱 테스트 2: 닫는 태그 없는 경우")
    print("=" * 80)

    template_no_close = """<SYSTEM>
시스템 메시지입니다.
</SYSTEM>

<USER>
사용자 메시지입니다.
</USER>

<ASSISTANT>
어시스턴트 메시지입니다.
"""  # </ASSISTANT> 없음

    messages = OllamaPromptBuilder._parse_chat_template(template_no_close)

    print(f"파싱된 메시지 수: {len(messages)}")
    for msg in messages:
        print(f"  - role: {msg['role']}")

    # 닫는 태그가 없으면 assistant가 파싱되지 않음
    roles = [msg["role"] for msg in messages]
    if "assistant" in roles:
        print("✓ assistant가 파싱됨")
    else:
        print("✗ assistant가 파싱되지 않음 (닫는 태그 필요)")


def test_actual_template_parsing():
    """실제 ver1.0.0 템플릿 파일 파싱 테스트"""
    print("\n" + "=" * 80)
    print("파싱 테스트 3: 실제 ver1.0.0 템플릿 파일")
    print("=" * 80)

    template_path = Path(__file__).parent.parent / "src" / "prompt" / "qwen3_2507_thinking_ver1.0.0.txt"

    if not template_path.exists():
        print(f"✗ 템플릿 파일 없음: {template_path}")
        return False

    template_content = template_path.read_text(encoding="utf-8")
    print(f"✓ 템플릿 파일: {template_path}")
    print(f"✓ 템플릿 길이: {len(template_content)} 문자")

    # 태그 존재 여부 확인
    has_system_open = "<SYSTEM>" in template_content
    has_system_close = "</SYSTEM>" in template_content
    has_user_open = "<USER>" in template_content
    has_user_close = "</USER>" in template_content
    has_assistant_open = "<ASSISTANT>" in template_content
    has_assistant_close = "</ASSISTANT>" in template_content

    print(f"\n태그 존재 여부:")
    print(f"  <SYSTEM>: {has_system_open}, </SYSTEM>: {has_system_close}")
    print(f"  <USER>: {has_user_open}, </USER>: {has_user_close}")
    print(f"  <ASSISTANT>: {has_assistant_open}, </ASSISTANT>: {has_assistant_close}")

    if has_assistant_open and not has_assistant_close:
        print("\n⚠️  경고: <ASSISTANT> 태그가 열렸지만 </ASSISTANT>로 닫히지 않음!")
        print("    → 정규식 패턴이 매칭되지 않아 assistant 메시지가 파싱되지 않습니다.")
        return False

    # 파싱 테스트
    messages = OllamaPromptBuilder._parse_chat_template(template_content)
    print(f"\n✓ 파싱된 메시지 수: {len(messages)}")
    for msg in messages:
        print(f"  - role: {msg['role']}, content 길이: {len(msg['content'])}")

    return True


# ============================================================================
# 파트 2: 빌더 및 구조 검증 테스트
# ============================================================================


def test_template_exists():
    """템플릿 파일 존재 여부 확인"""
    print("=" * 80)
    print("TEST 1: 템플릿 파일 존재 확인")
    print("=" * 80)

    template_path = Path(__file__).parent.parent / "src" / "prompt" / "qwen3_2507_thinking_ver1.0.0.txt"

    if template_path.exists():
        print(f"✓ 템플릿 파일 존재: {template_path}")
        content = template_path.read_text(encoding="utf-8")
        print(f"✓ 템플릿 길이: {len(content)} 문자")
        return True
    else:
        print(f"✗ 템플릿 파일 없음: {template_path}")
        return False


def test_builder_initialization():
    """빌더 초기화 테스트"""
    print("\n" + "=" * 80)
    print("TEST 2: 빌더 초기화 테스트")
    print("=" * 80)

    try:
        builder = OllamaPromptBuilder(template_name="qwen3_2507_thinking_ver1.0.0")
        print("✓ 빌더 초기화 성공")
        print(f"✓ 템플릿 이름: {builder.template_name}")
        print(f"✓ 템플릿 캐싱: {builder._template_cache is not None}")
        return builder
    except Exception as e:
        print(f"✗ 빌더 초기화 실패: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_message_structure(messages: List[Dict], expected_roles: List[str]) -> bool:
    """메시지 구조 검증"""
    actual_roles = [msg["role"] for msg in messages]

    if actual_roles == expected_roles:
        print(f"  ✓ 메시지 역할 순서 정상: {actual_roles}")
        return True
    else:
        print(f"  ✗ 메시지 역할 순서 비정상:")
        print(f"    예상: {expected_roles}")
        print(f"    실제: {actual_roles}")
        return False


def verify_n_choices_substitution(messages: List[Dict], expected_n: int) -> bool:
    """n_choices 변수가 올바르게 치환되었는지 확인"""
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    # "[선택지] (총 n개)" 패턴 확인
    if f"[선택지] (총 {expected_n}개)" in user_content:
        print(f"  ✓ n_choices 치환 성공: (총 {expected_n}개)")
        return True
    else:
        print(f"  ✗ n_choices 치환 실패")
        print(f"    예상: (총 {expected_n}개)")
        # 실제 내용 일부 출력
        if "[선택지]" in user_content:
            idx = user_content.find("[선택지]")
            snippet = user_content[idx : idx + 50]
            print(f"    실제: {snippet}...")
        return False


def verify_choices_format(messages: List[Dict], choices: List[str]) -> bool:
    """선택지 포맷이 올바른지 확인 (1. 형식)"""
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    all_found = True
    for idx, choice in enumerate(choices, start=1):
        expected_line = f"{idx}. {choice}"
        if expected_line in user_content:
            print(f"  ✓ 선택지 {idx} 포맷 정상: {expected_line}")
        else:
            print(f"  ✗ 선택지 {idx} 포맷 비정상: {expected_line}")
            all_found = False

    return all_found


def verify_question_plus(messages: List[Dict], question_plus: str = None) -> bool:
    """보기(question_plus) 포함 여부 확인"""
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    if question_plus:
        if "[보기]" in user_content and question_plus in user_content:
            print(f"  ✓ 보기 포함 확인됨")
            return True
        else:
            print(f"  ✗ 보기가 포함되지 않음")
            return False
    else:
        if "[보기]" not in user_content:
            print(f"  ✓ 보기 없음 확인됨 (정상)")
            return True
        else:
            print(f"  ✗ 보기가 있으면 안되는데 포함됨")
            return False


def verify_assistant_message(messages: List[Dict]) -> bool:
    """ASSISTANT 메시지가 올바르게 파싱되었는지 확인"""
    assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]

    if len(assistant_msgs) == 1:
        content = assistant_msgs[0]["content"]
        print(f"  ✓ ASSISTANT 메시지 존재")
        print(f"    내용: {content[:100]}...")

        # "n이하의 자연수 '단 한개만' 출력" 같은 키워드 확인
        if "단 한개만" in content or "자연수" in content:
            print(f"  ✓ ASSISTANT 메시지 내용 정상")
            return True
        else:
            print(f"  ⚠ ASSISTANT 메시지 내용이 예상과 다를 수 있음")
            return True
    else:
        print(f"  ✗ ASSISTANT 메시지 개수 비정상: {len(assistant_msgs)}개")
        return False


def test_4_choices():
    """선택지 4개 테스트"""
    print("\n" + "=" * 80)
    print("TEST 3: 선택지 4개 테스트")
    print("=" * 80)

    builder = OllamaPromptBuilder(template_name="qwen3_2507_thinking_ver1.0.0")

    test_data = {
        "paragraph": "조선시대에는 과전법, 직전법, 관수관급제 등 다양한 토지 제도가 시행되었다.",
        "question": "다음 중 조선시대 토지 제도가 아닌 것은?",
        "choices": ["과전법", "직전법", "녹봉제", "균전제"],
    }

    print(f"\n테스트 데이터:")
    print(f"  지문: {test_data['paragraph'][:50]}...")
    print(f"  질문: {test_data['question']}")
    print(f"  선택지: {test_data['choices']}")
    print(f"  선택지 개수: {len(test_data['choices'])}")

    messages = builder.build_single(**test_data)

    print(f"\n생성된 메시지:")
    print(f"  메시지 수: {len(messages)}")

    # 검증
    results = []
    results.append(verify_message_structure(messages, ["system", "user", "assistant"]))
    results.append(verify_n_choices_substitution(messages, 4))
    results.append(verify_choices_format(messages, test_data["choices"]))
    results.append(verify_question_plus(messages, None))
    results.append(verify_assistant_message(messages))

    if all(results):
        print("\n✓ 선택지 4개 테스트 전체 통과")
        return True
    else:
        print("\n✗ 선택지 4개 테스트 실패")
        return False


def test_5_choices():
    """선택지 5개 테스트"""
    print("\n" + "=" * 80)
    print("TEST 4: 선택지 5개 테스트")
    print("=" * 80)

    builder = OllamaPromptBuilder(template_name="qwen3_2507_thinking_ver1.0.0")

    test_data = {
        "paragraph": "대한제국은 1897년 고종이 선포한 국가이다.",
        "question": "다음 중 한국의 역사 시대가 아닌 것은?",
        "choices": ["고조선", "삼국시대", "고려", "조선", "대한제국"],
    }

    print(f"\n테스트 데이터:")
    print(f"  지문: {test_data['paragraph'][:50]}...")
    print(f"  질문: {test_data['question']}")
    print(f"  선택지: {test_data['choices']}")
    print(f"  선택지 개수: {len(test_data['choices'])}")

    messages = builder.build_single(**test_data)

    print(f"\n생성된 메시지:")
    print(f"  메시지 수: {len(messages)}")

    # 검증
    results = []
    results.append(verify_message_structure(messages, ["system", "user", "assistant"]))
    results.append(verify_n_choices_substitution(messages, 5))
    results.append(verify_choices_format(messages, test_data["choices"]))
    results.append(verify_question_plus(messages, None))
    results.append(verify_assistant_message(messages))

    if all(results):
        print("\n✓ 선택지 5개 테스트 전체 통과")
        return True
    else:
        print("\n✗ 선택지 5개 테스트 실패")
        return False


def test_with_question_plus():
    """보기(question_plus) 포함 테스트"""
    print("\n" + "=" * 80)
    print("TEST 5: 보기 포함 테스트 (선택지 4개)")
    print("=" * 80)

    builder = OllamaPromptBuilder(template_name="qwen3_2507_thinking_ver1.0.0")

    test_data = {
        "paragraph": "한국의 경제 성장에 대한 여러 견해가 있다.",
        "question": "다음 보기 중 옳은 것을 모두 고르면?",
        "choices": ["ㄱ, ㄴ", "ㄱ, ㄷ", "ㄴ, ㄷ", "ㄱ, ㄴ, ㄷ"],
        "question_plus": "ㄱ. 1960년대 경제개발 5개년 계획이 시작되었다.\nㄴ. 1970년대 중화학공업이 발전하였다.\nㄷ. 1980년대 3저 호황이 있었다.",
    }

    print(f"\n테스트 데이터:")
    print(f"  지문: {test_data['paragraph'][:50]}...")
    print(f"  질문: {test_data['question']}")
    print(f"  선택지: {test_data['choices']}")
    print(f"  선택지 개수: {len(test_data['choices'])}")
    print(f"  보기: {test_data['question_plus'][:50]}...")

    messages = builder.build_single(**test_data)

    print(f"\n생성된 메시지:")
    print(f"  메시지 수: {len(messages)}")

    # 검증
    results = []
    results.append(verify_message_structure(messages, ["system", "user", "assistant"]))
    results.append(verify_n_choices_substitution(messages, 4))
    results.append(verify_choices_format(messages, test_data["choices"]))
    results.append(verify_question_plus(messages, test_data["question_plus"]))
    results.append(verify_assistant_message(messages))

    if all(results):
        print("\n✓ 보기 포함 테스트 전체 통과")
        return True
    else:
        print("\n✗ 보기 포함 테스트 실패")
        return False


def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n" + "=" * 80)
    print("TEST 6: 엣지 케이스 테스트")
    print("=" * 80)

    builder = OllamaPromptBuilder(template_name="qwen3_2507_thinking_ver1.0.0")

    # 6-1: 선택지 2개 (최소)
    print("\n[6-1] 선택지 2개 (최소)")
    try:
        messages = builder.build_single(
            paragraph="테스트 지문",
            question="테스트 질문",
            choices=["선택지1", "선택지2"],
        )
        verify_n_choices_substitution(messages, 2)
        print("✓ 선택지 2개 테스트 통과")
    except Exception as e:
        print(f"✗ 선택지 2개 테스트 실패: {e}")

    # 6-2: 긴 지문
    print("\n[6-2] 긴 지문 테스트")
    long_paragraph = "테스트 " * 500  # 약 3000자
    try:
        messages = builder.build_single(
            paragraph=long_paragraph,
            question="테스트 질문",
            choices=["선택지1", "선택지2", "선택지3"],
        )
        print(f"✓ 긴 지문 테스트 통과 (지문 길이: {len(long_paragraph)} 문자)")
    except Exception as e:
        print(f"✗ 긴 지문 테스트 실패: {e}")

    # 6-3: 특수문자 포함
    print("\n[6-3] 특수문자 포함 테스트")
    try:
        messages = builder.build_single(
            paragraph='테스트 지문: <특수>, &문자, "따옴표"',
            question="테스트 질문 (괄호)",
            choices=["선택지-1", "선택지/2", "선택지@3"],
        )
        print("✓ 특수문자 포함 테스트 통과")
    except Exception as e:
        print(f"✗ 특수문자 포함 테스트 실패: {e}")

    # 6-4: 빈 question_plus
    print("\n[6-4] 빈 question_plus 테스트")
    try:
        messages = builder.build_single(
            paragraph="테스트 지문",
            question="테스트 질문",
            choices=["선택지1", "선택지2"],
            question_plus="",
        )
        verify_question_plus(messages, None)  # 빈 문자열은 None처럼 처리되어야 함
        print("✓ 빈 question_plus 테스트 통과")
    except Exception as e:
        print(f"✗ 빈 question_plus 테스트 실패: {e}")


def show_prompt_output_by_choice_count():
    """선택지 개수별(2~5개) 실제 프롬프트 출력 표시"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "선택지 개수별 실제 프롬프트 출력" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")

    builder = OllamaPromptBuilder(template_name="qwen3_2507_thinking_ver1.0.0")

    base_paragraph = "대한민국 임시정부는 1919년 3·1 운동 직후 상하이에서 수립되었다."
    base_question = "다음 중 대한민국 임시정부에 대한 설명으로 옳은 것은?"

    all_choices = [
        "1918년에 수립되었다.",
        "베이징에서 수립되었다.",
        "3·1 운동 이전에 수립되었다.",
        "1919년 상하이에서 수립되었다.",
        "일본에서 수립되었다.",
    ]

    # 선택지 2, 3, 4, 5개 케이스만 표시
    for n_choices in [2, 3, 4, 5]:
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + f" 선택지 {n_choices}개 케이스".center(76) + "║")
        print("╚" + "=" * 78 + "╝")

        choices = all_choices[:n_choices]

        messages = builder.build_single(
            paragraph=base_paragraph,
            question=base_question,
            choices=choices,
        )

        print(f"\n📊 메타 정보:")
        print(f"  - 총 메시지 수: {len(messages)}")
        print(f"  - 역할 순서: {' → '.join([msg['role'] for msg in messages])}")
        print(f"  - 선택지 개수: {n_choices}")

        for i, msg in enumerate(messages, start=1):
            print(f"\n{'━' * 80}")
            print(f"┃ Message {i} | Role: {msg['role'].upper()}")
            print(f"{'━' * 80}")
            content = msg["content"]

            # USER 메시지의 경우 선택지 부분을 하이라이트
            if msg["role"] == "user":
                # [선택지] 부분 찾기
                if "[선택지]" in content:
                    idx = content.find("[선택지]")
                    before = content[:idx]
                    after = content[idx:]

                    # 지문, 질문 부분 출력
                    if before:
                        print(before.rstrip())

                    # 선택지 부분 강조
                    print("\n" + "▼" * 40 + " [선택지] 섹션 시작 " + "▼" * 40)
                    print(after)
                    print("▲" * 40 + " [선택지] 섹션 끝 " + "▲" * 40)
                else:
                    print(content)
            else:
                # SYSTEM, ASSISTANT 메시지는 그대로 출력
                print(content)

        print("\n" + "─" * 80)

        # 검증 정보 추가
        user_msg = next((msg for msg in messages if msg["role"] == "user"), None)
        if user_msg:
            content = user_msg["content"]
            if f"[선택지] (총 {n_choices}개)" in content:
                print(f"✓ n_choices 변수 정상 치환: (총 {n_choices}개)")
            else:
                print(f"✗ n_choices 변수 치환 실패")

            # 각 선택지 포맷 확인
            all_found = True
            for idx in range(1, n_choices + 1):
                expected = f"{idx}. "
                if expected not in content:
                    all_found = False
                    break

            if all_found:
                print(f"✓ 선택지 포맷 정상: 1. ~ {n_choices}. 형식")
            else:
                print(f"✗ 선택지 포맷 비정상")

        print("=" * 80)


def main():
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "프롬프트 ver1.0.0 통합 테스트" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    # 파트 1: 파싱 기능 테스트
    print("\n" + "┏" + "━" * 78 + "┓")
    print("┃" + " " * 25 + "파트 1: 파싱 기능 테스트" + " " * 29 + "┃")
    print("┗" + "━" * 78 + "┛")

    test_parse_chat_template_basic()
    test_parse_chat_template_without_closing_tag()
    parsing_ok = test_actual_template_parsing()

    if not parsing_ok:
        print("\n❌ 템플릿 파싱에 문제가 있어 테스트를 중단합니다.")
        return

    # 파트 2: 구조 검증 테스트
    print("\n" + "┏" + "━" * 78 + "┓")
    print("┃" + " " * 25 + "파트 2: 구조 검증 테스트" + " " * 29 + "┃")
    print("┗" + "━" * 78 + "┛")

    results = []

    # 템플릿 존재 확인
    template_ok = test_template_exists()
    if not template_ok:
        print("\n❌ 템플릿 파일이 없어 테스트를 중단합니다.")
        return

    builder = test_builder_initialization()
    if not builder:
        print("\n❌ 빌더 초기화 실패로 테스트를 중단합니다.")
        return

    results.append(("선택지 4개", test_4_choices()))
    results.append(("선택지 5개", test_5_choices()))
    results.append(("보기 포함", test_with_question_plus()))

    # 파트 3: 엣지 케이스 (참고용, 결과에 미포함)
    print("\n" + "┏" + "━" * 78 + "┓")
    print("┃" + " " * 25 + "파트 3: 엣지 케이스 테스트" + " " * 28 + "┃")
    print("┗" + "━" * 78 + "┛")
    test_edge_cases()

    # 파트 4: 실제 프롬프트 출력 (선택지 2, 3, 4, 5개)
    print("\n" + "┏" + "━" * 78 + "┓")
    print("┃" + " " * 20 + "파트 4: 선택지별 실제 프롬프트 출력" + " " * 23 + "┃")
    print("┗" + "━" * 78 + "┛")
    show_prompt_output_by_choice_count()

    # 최종 결과
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "최종 결과" + " " * 39 + "║")
    print("╠" + "=" * 78 + "╣")

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"║  {test_name:<40} {status:>35} ║")

    print("╠" + "=" * 78 + "╣")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print(f"║  총 테스트: {total_tests}개                                                        ║")
    print(f"║  통과: {passed_tests}개                                                            ║")
    print(f"║  실패: {total_tests - passed_tests}개                                                            ║")

    if passed_tests == total_tests:
        print("║                                                                              ║")
        print("║  " + " " * 25 + "🎉 모든 테스트 통과! 🎉" + " " * 25 + "║")
    else:
        print("║                                                                              ║")
        print("║  " + " " * 27 + "⚠️  일부 테스트 실패" + " " * 28 + "║")

    print("╚" + "=" * 78 + "╝")


if __name__ == "__main__":
    main()
