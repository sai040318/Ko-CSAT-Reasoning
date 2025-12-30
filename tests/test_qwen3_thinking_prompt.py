import pytest
from pathlib import Path
from src.prompt.qwen3_2507_thinking_prompt import Qwen3ThinkingPromptBuilder


# ==================== Fixtures ====================

@pytest.fixture
def builder():
    """기본 빌더 인스턴스"""
    return Qwen3ThinkingPromptBuilder()


@pytest.fixture
def sample_single_data():
    """단일 문제 샘플 데이터"""
    return {
        "paragraph": "고종은 대한제국을 선포하였다.",
        "question": "다음 중 옳은 것은?",
        "choices": ["조선", "대한제국", "고려", "백제", "신라"],
    }


@pytest.fixture
def sample_batch_data():
    """배치 샘플 데이터"""
    return {
        "paragraph": [
            "고종은 대한제국을 선포하였다.",
            "세종대왕은 한글을 창제하였다.",
            "이순신 장군은 거북선을 만들었다.",
        ],
        "question": [
            "다음 중 옳은 것은?",
            "세종대왕의 업적은?",
            "이순신 장군과 관련된 것은?",
        ],
        "choices": [
            ["조선", "대한제국", "고려", "백제", "신라"],
            ["훈민정음", "팔만대장경", "직지심체요절", "무구정광대다라니경", "삼국사기"],
            ["거북선", "판옥선", "조운선", "황포돛배", "멍텅구리배"],
        ],
        "question_plus": [None, "ㄱ. 1443년\nㄴ. 1446년", None],
        "answer": [2, 1, 1],
    }


# ==================== 기본 기능 테스트 ====================

class TestBuildSingleBasic:
    """build_single 기본 기능 테스트"""

    def test_returns_list_of_dicts(self, builder, sample_single_data):
        """반환 타입 확인"""
        messages = builder.build_single(**sample_single_data)

        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)

    def test_message_structure(self, builder, sample_single_data):
        """메시지 구조 확인 (role, content 키 존재)"""
        messages = builder.build_single(**sample_single_data)

        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_default_roles(self, builder, sample_single_data):
        """기본 역할 순서: system -> user"""
        messages = builder.build_single(**sample_single_data)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_with_answer_adds_assistant(self, builder, sample_single_data):
        """answer 제공 시 assistant 역할 추가"""
        messages = builder.build_single(**sample_single_data, answer=2)

        assert len(messages) == 3
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "2"


# ==================== 파라미터화 테스트 ====================

class TestParameterized:
    """다양한 입력값에 대한 파라미터화 테스트"""

    @pytest.mark.parametrize("answer", [1, 2, 3, 4, 5])
    def test_all_valid_answers(self, builder, sample_single_data, answer):
        """모든 유효한 정답 번호 테스트 (1-5)"""
        messages = builder.build_single(**sample_single_data, answer=answer)

        assert messages[-1]["content"] == str(answer)

    @pytest.mark.parametrize("num_choices", [2, 3, 4, 5, 6, 7])
    def test_various_choice_counts(self, builder, num_choices):
        """다양한 선택지 개수 테스트"""
        choices = [f"선택지{i}" for i in range(1, num_choices + 1)]
        messages = builder.build_single(
            paragraph="지문",
            question="질문",
            choices=choices,
        )

        user_content = messages[1]["content"]
        for i, choice in enumerate(choices, 1):
            assert f"{i} - {choice}" in user_content

    @pytest.mark.parametrize("question_plus,expected_in_content", [
        ("ㄱ. 보기1", True),
        ("", False),
        (None, False),
    ])
    def test_question_plus_variations(self, builder, question_plus, expected_in_content):
        """question_plus 다양한 값 테스트"""
        messages = builder.build_single(
            paragraph="지문",
            question="질문",
            choices=["A", "B"],
            question_plus=question_plus,
        )

        user_content = messages[1]["content"]
        if expected_in_content:
            assert "<보 기>" in user_content
        else:
            assert "<보 기>" not in user_content

    @pytest.mark.parametrize("paragraph", [
        "짧은 지문",
        "중간 길이의 지문입니다. 여러 문장이 포함될 수 있습니다.",
        "매우 긴 지문입니다. " * 50,
        "특수문자 포함: !@#$%^&*()",
        "줄바꿈\n포함\n지문",
        "한글English混合지문",
    ])
    def test_various_paragraph_formats(self, builder, paragraph):
        """다양한 지문 형식 테스트"""
        messages = builder.build_single(
            paragraph=paragraph,
            question="질문",
            choices=["A", "B"],
        )

        assert paragraph in messages[1]["content"]


# ==================== 배치 처리 테스트 ====================

class TestBuildBatch:
    """build_batch 기능 테스트"""

    def test_batch_length(self, builder, sample_batch_data):
        """배치 결과 길이 확인"""
        result = builder.build_batch(sample_batch_data)

        assert len(result) == len(sample_batch_data["paragraph"])

    def test_batch_each_item_valid(self, builder, sample_batch_data):
        """배치 각 항목 유효성 확인"""
        result = builder.build_batch(sample_batch_data)

        for messages in result:
            assert len(messages) >= 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    def test_batch_without_optional_fields(self, builder):
        """선택 필드 없이 배치 처리"""
        examples = {
            "paragraph": ["지문1", "지문2"],
            "question": ["질문1", "질문2"],
            "choices": [["A", "B"], ["C", "D"]],
        }

        result = builder.build_batch(examples)

        assert len(result) == 2
        assert all(len(msgs) == 2 for msgs in result)

    def test_batch_mixed_answers(self, builder):
        """일부만 answer가 있는 배치"""
        examples = {
            "paragraph": ["지문1", "지문2", "지문3"],
            "question": ["질문1", "질문2", "질문3"],
            "choices": [["A", "B"], ["C", "D"], ["E", "F"]],
            "answer": [1, None, 2],
        }

        result = builder.build_batch(examples)

        assert len(result[0]) == 3  # answer 있음
        assert len(result[1]) == 2  # answer 없음
        assert len(result[2]) == 3  # answer 있음

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50, 100])
    def test_various_batch_sizes(self, builder, batch_size):
        """다양한 배치 크기 테스트"""
        examples = {
            "paragraph": [f"지문{i}" for i in range(batch_size)],
            "question": [f"질문{i}" for i in range(batch_size)],
            "choices": [["A", "B", "C"] for _ in range(batch_size)],
        }

        result = builder.build_batch(examples)

        assert len(result) == batch_size


# ==================== 포맷팅 테스트 ====================

class TestFormatting:
    """포맷팅 관련 테스트"""

    def test_choices_numbering(self, builder):
        """선택지 번호 매기기 확인"""
        messages = builder.build_single(
            paragraph="지문",
            question="질문",
            choices=["첫번째", "두번째", "세번째", "네번째", "다섯번째"],
        )

        content = messages[1]["content"]
        assert "1 - 첫번째" in content
        assert "2 - 두번째" in content
        assert "3 - 세번째" in content
        assert "4 - 네번째" in content
        assert "5 - 다섯번째" in content

    def test_question_plus_format(self, builder):
        """보기 포맷 확인"""
        messages = builder.build_single(
            paragraph="지문",
            question="질문",
            choices=["A", "B"],
            question_plus="ㄱ. 항목1\nㄴ. 항목2",
        )

        content = messages[1]["content"]
        assert "<보 기>" in content
        assert "ㄱ. 항목1" in content
        assert "ㄴ. 항목2" in content

    def test_system_message_content(self, builder, sample_single_data):
        """시스템 메시지 내용 확인"""
        messages = builder.build_single(**sample_single_data)

        system_content = messages[0]["content"]
        assert "숫자 하나" in system_content or "정답" in system_content

    def test_user_message_contains_all_parts(self, builder):
        """user 메시지에 모든 구성요소 포함 확인"""
        messages = builder.build_single(
            paragraph="테스트 지문입니다.",
            question="테스트 질문입니다?",
            choices=["선택A", "선택B"],
            question_plus="보기 내용",
        )

        content = messages[1]["content"]
        assert "테스트 지문입니다." in content
        assert "테스트 질문입니다?" in content
        assert "선택A" in content
        assert "선택B" in content
        assert "보기 내용" in content


# ==================== 엣지 케이스 테스트 ====================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_string_answer(self, builder, sample_single_data):
        """빈 문자열 answer 처리"""
        messages = builder.build_single(**sample_single_data, answer="")

        assert len(messages) == 2  # assistant 없음

    def test_single_choice(self, builder):
        """선택지 1개만 있는 경우"""
        messages = builder.build_single(
            paragraph="지문",
            question="질문",
            choices=["유일한 선택지"],
        )

        assert "1 - 유일한 선택지" in messages[1]["content"]

    def test_special_characters_in_content(self, builder):
        """특수문자 포함 테스트"""
        messages = builder.build_single(
            paragraph="지문에 {중괄호}와 <꺾쇠>가 포함",
            question="질문에 $달러$ 기호?",
            choices=["선택지 #1", "선택지 @2"],
        )

        content = messages[1]["content"]
        # format()에서 {중괄호}가 문제될 수 있으므로 확인
        assert "지문에" in content
        assert "질문에" in content

    def test_unicode_content(self, builder):
        """유니코드 콘텐츠 테스트"""
        messages = builder.build_single(
            paragraph="日本語テスト 中文测试 한글테스트",
            question="émoji 🎉 질문?",
            choices=["α", "β", "γ"],
        )

        content = messages[1]["content"]
        assert "日本語テスト" in content
        assert "🎉" in content

    def test_multiline_paragraph(self, builder):
        """여러 줄 지문 테스트"""
        paragraph = """첫 번째 문단입니다.

두 번째 문단입니다.
여러 줄로 구성되어 있습니다.

세 번째 문단입니다."""

        messages = builder.build_single(
            paragraph=paragraph,
            question="질문",
            choices=["A", "B"],
        )

        assert "첫 번째 문단" in messages[1]["content"]
        assert "세 번째 문단" in messages[1]["content"]

    def test_very_long_choices(self, builder):
        """매우 긴 선택지 테스트"""
        long_choice = "매우 긴 선택지입니다. " * 20
        messages = builder.build_single(
            paragraph="지문",
            question="질문",
            choices=[long_choice, "짧은 선택지"],
        )

        assert long_choice in messages[1]["content"]


# ==================== 템플릿 관련 테스트 ====================

class TestTemplate:
    """템플릿 로딩 및 캐싱 테스트"""

    def test_template_caching(self, builder):
        """템플릿 캐싱 동작 확인"""
        _ = builder.template
        first_cache = builder._template_cache

        _ = builder.template
        second_cache = builder._template_cache

        assert first_cache is second_cache

    def test_template_not_empty(self, builder):
        """템플릿이 비어있지 않음 확인"""
        assert builder.template
        assert len(builder.template) > 0

    def test_template_contains_placeholders(self, builder):
        """템플릿에 필수 플레이스홀더 포함 확인"""
        template = builder.template

        assert "{paragraph}" in template
        assert "{question}" in template
        assert "{choices}" in template

    def test_invalid_template_name(self):
        """존재하지 않는 템플릿 이름 에러"""
        builder = Qwen3ThinkingPromptBuilder(template_name="nonexistent_template")

        with pytest.raises(ValueError, match="Template not found"):
            _ = builder.template

    def test_custom_template_name(self):
        """커스텀 템플릿 이름 설정"""
        builder = Qwen3ThinkingPromptBuilder(template_name="qwen3_2507_thinking")

        assert builder.template_name == "qwen3_2507_thinking"


# ==================== 하위 호환성 테스트 ====================

class TestLegacyFunctions:
    """하위 호환성 함수 테스트"""

    def test_load_template_function(self):
        """load_template 함수 동작 확인"""
        from src.prompt.qwen3_2507_thinking_prompt import load_template

        template = load_template("qwen3_2507_thinking")

        assert template
        assert "{paragraph}" in template

    def test_parse_chat_template_function(self):
        """parse_chat_template 함수 동작 확인"""
        from src.prompt.qwen3_2507_thinking_prompt import parse_chat_template

        text = "<SYSTEM>시스템 메시지</SYSTEM><USER>유저 메시지</USER>"
        messages = parse_chat_template(text)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_chat_messages_function(self):
        """build_chat_messages 함수 동작 확인"""
        from src.prompt.qwen3_2507_thinking_prompt import build_chat_messages

        examples = {
            "paragraph": ["지문"],
            "question": ["질문"],
            "choices": [["A", "B"]],
            "question_plus": [None],
            "answer": [1],
        }

        result = build_chat_messages(
            template_name="qwen3_2507_thinking",
            examples=examples,
        )

        assert len(result) == 1
        assert len(result[0]) == 3


# ==================== 통합 테스트 ====================

class TestIntegration:
    """통합 테스트"""

    def test_full_workflow_single(self, builder):
        """단일 문제 전체 워크플로우"""
        messages = builder.build_single(
            paragraph="조선 후기 실학자 정약용은 목민심서를 저술하였다.",
            question="정약용의 저서로 옳은 것은?",
            choices=["삼국사기", "목민심서", "동의보감", "훈민정음", "직지심체요절"],
            question_plus="ㄱ. 조선 후기\nㄴ. 실학 사상",
            answer=2,
        )

        # 구조 확인
        assert len(messages) == 3

        # 역할 확인
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]

        # 내용 확인
        assert "정약용" in messages[1]["content"]
        assert "목민심서" in messages[1]["content"]
        assert "<보 기>" in messages[1]["content"]
        assert messages[2]["content"] == "2"

    def test_full_workflow_batch(self, builder, sample_batch_data):
        """배치 전체 워크플로우"""
        result = builder.build_batch(sample_batch_data)

        # 길이 확인
        assert len(result) == 3

        # 각 항목 구조 확인
        for i, messages in enumerate(result):
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

            # answer가 있으면 assistant 포함
            if sample_batch_data["answer"][i]:
                assert messages[-1]["role"] == "assistant"

    def test_ollama_compatible_format(self, builder, sample_single_data):
        """Ollama chat API 호환 포맷 확인"""
        messages = builder.build_single(**sample_single_data)

        # Ollama가 기대하는 형식 확인
        for msg in messages:
            assert set(msg.keys()) == {"role", "content"}
            assert msg["role"] in ["system", "user", "assistant"]
            assert isinstance(msg["content"], str)


# ==================== 성능 테스트 (선택적) ====================

@pytest.mark.slow
class TestPerformance:
    """성능 관련 테스트 (느린 테스트)"""

    def test_large_batch_performance(self, builder):
        """대용량 배치 처리"""
        batch_size = 1000
        examples = {
            "paragraph": [f"지문{i}" for i in range(batch_size)],
            "question": [f"질문{i}" for i in range(batch_size)],
            "choices": [["A", "B", "C", "D", "E"] for _ in range(batch_size)],
            "answer": [i % 5 + 1 for i in range(batch_size)],
        }

        result = builder.build_batch(examples)

        assert len(result) == batch_size

    def test_repeated_single_calls(self, builder, sample_single_data):
        """반복 호출 성능"""
        for _ in range(100):
            messages = builder.build_single(**sample_single_data)
            assert len(messages) == 2
