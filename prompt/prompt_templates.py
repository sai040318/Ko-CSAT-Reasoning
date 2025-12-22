from pathlib import Path
import re

TEMPLATE_DIR = Path(__file__).parent


def load_template(name: str) -> str:
    path = TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"Template not found: {name}")
    return path.read_text(encoding="utf-8")


def parse_chat_template(text: str) -> list[dict]:
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


def build_chat_messages(*, template_name: str, examples: dict) -> list[list[dict]]:
    template = load_template(template_name)
    chat_messages = []

    for p, q, qp, c, a in zip(
        examples["paragraph"],
        examples.get("question_plus", [None] * len(examples["paragraph"])),
        examples["question"],
        examples["choices"],
        examples["answer"],
    ):
        choices_str = "\n".join(
            f"{i+1} - {choice}" for i, choice in enumerate(c)
        )

        filled = template.format(
            paragraph=p,
            question_plus=qp or "",
            question=q,
            choices=choices_str,
        )

        messages = parse_chat_template(filled)

        # SFT용: assistant 정답은 코드에서만 붙임
        if a is not None:
            messages.append(
                {"role": "assistant", "content": str(a)}
            )

        chat_messages.append(messages)

    return chat_messages

if __name__ == "__main__":
    examples = {
        "paragraph": ["고종은 대한제국을 선포하였다."],
        "question_plus": [None],
        "question": ["다음 중 옳은 것은?"],
        "choices": [["조선", "대한제국", "고려"]],
        "answer": [2],
    }

    messages = build_chat_messages(
        template_name="base",
        examples=examples,
    )

    from pprint import pprint
    pprint(messages)