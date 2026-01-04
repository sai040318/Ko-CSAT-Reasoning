# (2) PromptBuilder: 선택지 개수(n_choices)와 보기 구획을 안정적으로 넣기
# src/prompt/ollama_prompt.py

from pathlib import Path
import re
from typing import Optional


# TODO: json으로 답하라는 형식 지정 프롬프트는 무조건 없애는 게 좋을듯 차라리 logit 쓰거나 thinking
# ollama api에 대해 더 공부할 것
class OllamaPromptBuilder:
    """
    Qwen3-2507-ollama 모델용 프롬프트 빌더 클래스.

    사용 예시:
        builder = OllamaPromptBuilder()
        messages = builder.build_single(
            paragraph="고종은 대한제국을 선포하였다.",
            question="다음 중 옳은 것은?",
            choices=["조선", "대한제국", "고려"],
        )
    """

    TEMPLATE_DIR = Path(__file__).parent

    def __init__(self, template_name: Optional[str] = None):
        """
        Args:
            template_name: 사용할 템플릿 이름 (기본값: qwen3-2507_base)
        """
        self.template_name = template_name
        if self.template_name is None:
            raise ValueError("template_name must be provided")
        self._template_cache: Optional[str] = None

    @property
    def template(self) -> str:
        """템플릿 내용을 캐싱하여 반환"""
        if self._template_cache is None:
            self._template_cache = self._load_template(self.template_name)
        return self._template_cache

    def _load_template(self, name: str) -> str:
        """템플릿 파일 로드"""
        path = self.TEMPLATE_DIR / f"{name}.txt"
        if not path.exists():
            raise ValueError(f"Template not found: {name}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _parse_chat_template(text: str) -> list[dict]:
        """
        <SYSTEM>...</SYSTEM>, <USER>...</USER> 등을
        [{"role": "...", "content": "..."}] 형태로 변환
        """
        messages = []
        pattern = re.compile(r"<(SYSTEM|USER|ASSISTANT)>(.*?)</\1>", re.S)

        for role, content in pattern.findall(text):
            messages.append(
                {
                    "role": role.lower(),
                    "content": content.strip(),
                }
            )
        return messages

    def _format_choices(self, choices: list[str]) -> str:
        """선택지를 포맷팅"""
        # 모델이 "1 - ..."에서 하이픈을 내용의 일부로 오인하는 경우가 있어 (.) 형식을 권장
        return "\n".join(f"{idx + 1}. {choice}" for idx, choice in enumerate(choices))

    def _format_question_plus(self, question_plus: Optional[str]) -> str:
        """보기 포맷팅"""
        if question_plus:
            return f"[보기]\n{question_plus}"
        return ""  # 템플릿에서 {question_plus}가 비면 해당 블록이 자연스럽게 사라지도록 설계

    # TODO MODELFILE, TEMPLATE 등으로 더 개선
    def build_single(
        self,
        paragraph: str,
        question: str,
        choices: list[str],
        question_plus: Optional[str] = None,
        answer: Optional[int] = None,
    ) -> list[dict]:
        """
        단일 문제에 대한 chat message 생성

        Args:
            paragraph: 지문
            question: 질문
            choices: 선택지 리스트
            question_plus: 보기 (선택)
            answer: 정답 번호 (선택, 학습 시 사용)

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        choices_str = self._format_choices(choices)
        qp_str = self._format_question_plus(question_plus)

        filled = self.template.format(
            paragraph=paragraph,
            question=question,
            question_plus=qp_str,
            choices=choices_str,
            n_choices=len(choices),
        )

        messages = self._parse_chat_template(filled)

        if answer is not None and answer != "":
            messages.append({"role": "assistant", "content": str(answer)})

        return messages

    def build_batch(self, examples: dict) -> list[list[dict]]:
        """
        배치 데이터에 대한 chat messages 생성

        Args:
            examples: {
                "paragraph": [...],
                "question": [...],
                "choices": [...],
                "question_plus": [...],  # optional
                "answer": [...]  # optional
            }

        Returns:
            [[{"role": "...", "content": "..."}, ...], ...]
        """
        n = len(examples["paragraph"])
        q_plus_list = examples.get("question_plus", [None] * n)
        answer_list = examples.get("answer", [None] * n)

        chat_messages = []
        for i in range(n):
            messages = self.build_single(
                paragraph=examples["paragraph"][i],
                question=examples["question"][i],
                choices=examples["choices"][i],
                question_plus=q_plus_list[i] if q_plus_list[i] else None,
                answer=answer_list[i] if answer_list[i] not in [None, ""] else None,
            )
            chat_messages.append(messages)

        return chat_messages


if __name__ == "__main__":
    from pprint import pprint
    import sys

    print("=" * 60)
    print("OllamaPromptBuilder 테스트")
    print("=" * 60)

    # 1. 템플릿 설정
    print("\n[1] 템플릿 확인")
    template_name = "qwen3_2507_base"
    template_dir = Path(__file__).parent
    print(f"템플릿 디렉토리: {template_dir}")
    print(f"테스트 템플릿: {template_name}")

    # 2. 빌더 초기화 테스트
    print("\n[2] 빌더 초기화 테스트")
    try:
        builder = OllamaPromptBuilder(template_name=template_name)
        print(f"✓ 빌더 초기화 성공: {builder.template_name}")
        print(f"✓ 템플릿 로드 성공 (길이: {len(builder.template)} 문자)")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

    # 3. 단일 문제 테스트 (question_plus 없음)
    print("\n[3] 단일 문제 테스트 (보기 없음)")
    try:
        single_msg = builder.build_single(
            paragraph="고종은 1897년에 국호를 대한제국으로 바꾸고 황제에 즉위하였다.",
            question="다음 중 고종이 선포한 국가는?",
            choices=["조선", "대한제국", "고려", "신라", "백제"],
            answer=2,
        )
        print("✓ 메시지 생성 성공")
        print(f"  - 메시지 수: {len(single_msg)}")
        print(f"  - 역할 순서: {[msg['role'] for msg in single_msg]}")
        print("\n메시지 내용:")
        pprint(single_msg, width=100)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()

    # 4. 단일 문제 테스트 (question_plus 있음)
    print("\n[4] 단일 문제 테스트 (보기 있음)")
    try:
        single_msg_plus = builder.build_single(
            paragraph="조선 시대의 토지 제도에 대한 지문입니다.",
            question="다음 <보기>의 설명 중 옳은 것을 모두 고르면?",
            choices=["ㄱ, ㄴ", "ㄱ, ㄷ", "ㄴ, ㄷ", "ㄱ, ㄴ, ㄷ"],
            question_plus="ㄱ. 과전법은 토지를 지급하였다.\nㄴ. 직전법은 현직 관리에게만 지급하였다.\nㄷ. 녹봉제는 토지 대신 곡물로 지급하였다.",
            answer=4,
        )
        print("✓ 메시지 생성 성공 (보기 포함)")
        print(f"  - 메시지 수: {len(single_msg_plus)}")
        print(f"\n마지막 user 메시지 미리보기:")
        user_msgs = [msg for msg in single_msg_plus if msg["role"] == "user"]
        if user_msgs:
            print(user_msgs[-1]["content"][:300] + "...")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # 5. 배치 테스트
    print("\n[5] 배치 처리 테스트")
    try:
        examples = {
            "paragraph": [
                "고종은 1897년에 국호를 대한제국으로 바꾸고 황제에 즉위하였다.",
                "세종대왕은 한글을 창제하였다.",
            ],
            "question": [
                "다음 중 고종이 선포한 국가는?",
                "세종대왕이 창제한 것은?",
            ],
            "choices": [
                ["조선", "대한제국", "고려", "신라", "백제"],
                ["한자", "한글", "영어", "일본어"],
            ],
            "question_plus": [None, None],
            "answer": [2, 2],
        }

        batch_msgs = builder.build_batch(examples)
        print(f"✓ 배치 처리 성공: {len(batch_msgs)}개 문제")
        for i, msgs in enumerate(batch_msgs):
            print(f"  - 문제 {i + 1}: {len(msgs)}개 메시지, 역할: {[m['role'] for m in msgs]}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()

    # 6. 선택지 포맷 테스트
    print("\n[6] 선택지 포맷팅 테스트")
    test_choices = ["선택지 1", "선택지 2", "선택지 3"]
    formatted = builder._format_choices(test_choices)
    print("입력:", test_choices)
    print(f"출력:\n{formatted}")

    # 7. 보기 포맷팅 테스트
    print("\n[7] 보기 포맷팅 테스트")
    test_qp = "ㄱ. 첫 번째\nㄴ. 두 번째"
    formatted_qp = builder._format_question_plus(test_qp)
    print("입력:", test_qp)
    print(f"출력:\n{formatted_qp}")

    print("\n" + "=" * 60)
    print("✓ 모든 테스트 완료")
    print("=" * 60)
