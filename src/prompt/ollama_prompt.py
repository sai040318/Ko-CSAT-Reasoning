from pathlib import Path
import re
from typing import Optional


class OllamaPromptBuilder:
    """
    Qwen3-2507 Thinking 모델용 프롬프트 빌더 클래스.

    사용 예시:
        builder = OllamaPromptBuilder()
        messages = builder.build_single(
            paragraph="고종은 대한제국을 선포하였다.",
            question="다음 중 옳은 것은?",
            choices=["조선", "대한제국", "고려"],
        )
    """

    TEMPLATE_DIR = Path(__file__).parent
    DEFAULT_TEMPLATE = "qwen3_2507_thinking"

    def __init__(self, template_name: Optional[str] = None):
        """
        Args:
            template_name: 사용할 템플릿 이름 (기본값: qwen3-2507_thinking)
        """
        self.template_name = template_name or self.DEFAULT_TEMPLATE
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
        return "\n".join(f"{idx + 1} - {choice}" for idx, choice in enumerate(choices))

    def _format_question_plus(self, question_plus: Optional[str]) -> str:
        """보기 포맷팅"""
        if question_plus:
            return f"<보 기>\n{question_plus}"
        return ""

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


# 하위 호환성을 위한 함수 래퍼
def load_template(name: str) -> str:
    builder = OllamaPromptBuilder(template_name=name)
    return builder._load_template(name)


def parse_chat_template(text: str) -> list[dict]:
    return OllamaPromptBuilder._parse_chat_template(text)


def build_chat_messages(*, template_name: str, examples: dict) -> list[list[dict]]:
    builder = OllamaPromptBuilder(template_name=template_name)
    return builder.build_batch(examples)


if __name__ == "__main__":
    # 단일 문제 테스트
    builder = OllamaPromptBuilder()

    single_msg = builder.build_single(
        paragraph="고종은 대한제국을 선포하였다.",
        question="다음 중 옳은 것은?",
        choices=["조선", "대한제국", "고려"],
        answer=2,
    )

    print("=== Single Message ===")
    from pprint import pprint

    pprint(single_msg)

    # 배치 테스트
    examples = {
        "paragraph": ["고종은 대한제국을 선포하였다."],
        "question_plus": [None],
        "question": ["다음 중 옳은 것은?"],
        "choices": [["조선", "대한제국", "고려"]],
        "answer": [2],
    }

    batch_msgs = builder.build_batch(examples)

    print("\n=== Batch Messages ===")
    pprint(batch_msgs)
