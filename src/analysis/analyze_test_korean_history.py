import os
import ast
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


def parse_choices(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            try:
                return ast.literal_eval(raw)
            except Exception:
                return [raw]
    return []


def classify_with_gpt4(client: OpenAI, paragraph: str, question: str, choices, question_plus: str):
    choices_str = "\n".join(f"- {c}" for c in choices) if choices else "없음"
    prompt = f"""
    You are an expert classifier for a Korean History Exam RAG system.
    Classify whether the problem requires **External Knowledge Retrieval (Label: A)** or is a **Reading Comprehension task (Label: B)**.

    ### [CRITICAL RULES]

    **1. LENGTH HEURISTIC (Strong Indicator for B)**
    - If the passage is **very long** (e.g., multiple paragraphs, resembles a CSAT non-fiction/essay), it is almost certainly **Label B**.
    - Long passages usually contain all the logic/answers within the text. Do NOT search just because it mentions historical figures.

    **2. ENTITY HEURISTIC (Strong Indicator for A)**
    - If a **short/medium** passage uses placeholders like **"(가)", "(나)", "This King", "The organization"** without naming them, it is **Label A**.
    - You must search to identify who "(가)" is.

    ---

    ### [Detailed Classification Criteria]

    **Label A: Retrieval Required (Korean History Knowledge)**
    - **Identification:** The passage uses "(가)", "Underlined King", "This country" but does not explicitly name them. You need external knowledge to identify the entity.
    - **Fact verification:** The question asks for chronological order, specific dates, or "other achievements" not mentioned in the text.
    - **Source Material:** The text is a raw historical record (e.g., Samguk Sagi, Joseon Annals) written in an archaic style, and the question asks for background facts.

    **Label B: No Retrieval Needed (Reading Comprehension)**
    - **Long Expository Text:** A long explanation or essay about a historical topic (e.g., Silhak, Confucianism analysis). The answer is based on "consistency with the text" or "author's argument".
    - **Verbatim Match:** The answer options are explicitly stated or paraphrased in the passage.
    - **General Topics:** Science, Geography, Ethics, or general Social Studies.

    ### [Few-Shot Examples]

    **Case 1 (Long Text / Reading Comp -> B)**
    Input:
    - Passage: "(Long text about 18th-century Northern Learning)... Park Je-ga argued that consumption stimulates production... (continuing for 10+ lines)..."
    - Question: "Which statement is NOT consistent with the passage?"
    - Choices: "1. Park Je-ga emphasized consumption..."
    Output: {{ "reason": "The passage is a long expository text. The question asks for consistency with the provided text. All answers can be found by reading.", "label": "B" }}

    **Case 2 (Placeholder Heuristic -> A)**
    Input:
    - Passage: "(Ga) established the Gwageo system to weaken the noble families."
    - Question: "What is the correct description of the king (Ga)?"
    - Choices: "1. Enacted the Slave Review Act."
    Output: {{ "reason": "The passage uses the placeholder (Ga). To answer, one must identify (Ga) as King Gwangjong using external knowledge.", "label": "A" }}

    **Case 3 (Specific Fact -> A)**
    Input:
    - Passage: "The army retreated from Wihwado."
    - Question: "What happened immediately AFTER this event?"
    - Choices: "1. The Joseon Dynasty was founded."
    Output: {{ "reason": "The question asks for chronological order/subsequent events not described in the text. External history knowledge is required.", "label": "A" }}

    ### [Target Problem]
    Passage: {paragraph[:1500]} 
    Question: {question}
    Question Plus: {question_plus or 'None'}
    Choices: {choices_str}

    **Output Format:** JSON only. {{"label": "A" or "B" }}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        # 실패 시 보수적으로 B(검색 안함)로 처리
        return {"label": "B"}


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수를 설정하세요.")
    client = OpenAI(api_key=api_key)

    df_test = pd.read_csv("src/data/test.csv")
    results = []

    print("🚀 GPT-4o 분류 시작...")
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        paragraph = row.get("paragraph", "")
        question = row.get("question", "")
        choices = parse_choices(row.get("choices", []))
        question_plus = row.get("question_plus", "")

        cls_result = classify_with_gpt4(client, paragraph, question, choices, question_plus)
        results.append(
            {
                "id": row.get("id", ""),
                "label": cls_result.get("label", "B"),
            }
        )

    df_cls = pd.DataFrame(results)
    df_cls.to_csv("gpt4_classification_results1.csv", index=False)
    print("✅ 분류 완료! gpt4_classification_results.csv 저장됨")
