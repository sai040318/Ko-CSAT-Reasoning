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
    """
    zip 없이 row 단위로 순회
    examples: {"paragraph": [...], "question": [...], "question_plus": [...], "choices": [...], "answer": [...]}
    """
    template = load_template(template_name)
    chat_messages = []

    n = len(examples["paragraph"])
    # 존재하지 않는 컬럼은 기본값으로 채움
    q_plus_list = examples.get("question_plus", [""] * n)


    for i in range(n):
        p = examples["paragraph"][i]
        q = examples["question"][i]  # 항상 존재
        qp_raw = q_plus_list[i]
        qp = f"<보 기>\n{qp_raw}" if qp_raw else ""
        c = examples["choices"][i]
        d = examples["documents"][i] or ""
        a = examples["answer"][i]


        choices_str = "\n".join(f"{idx+1} - {choice}" for idx, choice in enumerate(c))

        filled = template.format(
            paragraph=p,
            question_plus=qp,
            question=q,
            choices=choices_str,
            documents=d
        )

        messages = parse_chat_template(filled)

        if a not in [None, ""]:
            messages.append({"role": "assistant", "content": str(a)})
        
        print(messages)

        chat_messages.append(messages)

    return chat_messages


if __name__ == "__main__":
    examples = {
        "paragraph": ["고종은 대한제국을 선포하였다."],
        "question_plus": [None],
        "question": ["다음 중 옳은 것은?"],
        "choices": [["조선", "대한제국", "고려"]],
        "documents": [None],
        "answer": [2],
    }

    messages = build_chat_messages(
        template_name="base",
        examples=examples,
    )

    from pprint import pprint
    pprint(messages)
