# src/analysis/analyze_test_korean_history.py

import os
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TEST_PATH = "src/data/test.csv"
OUTPUT_PATH = "analysis/test_korean_history_structure.json"
MODEL_NAME = "gpt-4.1-mini"

SYSTEM_PROMPT = """
너는 수능 객관식 문제의 과목과 형식만 분류하는 분석 도우미이다.
정답을 추론하거나 문제를 풀이하지 않는다.
문제의 구조적 특징만 분류한다.
"""

USER_PROMPT_TEMPLATE = """
다음은 수능 객관식 문제이다.

[지문]
{paragraph}

[문제]
{question}

[선지]
{choices}

❗주의사항:
- 정답을 추론하거나 풀이하지 마라.
- 문제의 의미를 설명하지 마라.

아래 항목만 JSON으로 출력하라:
1. is_korean_history: 한국사 문제 여부 (true / false)
2. problem_type: 한국사 문제라면 아래 중 하나, 아니면 null
   ["사료 해석형", "사실 판단형", "사건 순서형", "비교/대조형", "기타"]
3. has_marker: (가),(나),(다) 같은 사료 지시어 존재 여부 (true / false)
4. paragraph_length: ["짧음", "중간", "김"]
"""

def main():
    df = pd.read_csv(TEST_PATH)

    total = len(df)
    korean_history_count = 0

    type_counter = Counter()
    marker_counter = Counter()
    length_counter = Counter()

    per_problem = []

    for _, row in tqdm(df.iterrows(), total=total):
        prompt = USER_PROMPT_TEMPLATE.format(
            paragraph=row.get("paragraph", ""),
            question=row.get("question", ""),
            choices=row.get("choices", "")
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        content = response.choices[0].message.content

        try:
            analysis = json.loads(content)
        except json.JSONDecodeError:
            continue

        if analysis.get("is_korean_history"):
            korean_history_count += 1

            ptype = analysis.get("problem_type", "기타")
            type_counter[ptype] += 1

            marker_counter["has_marker" if analysis.get("has_marker") else "no_marker"] += 1
            length_counter[analysis.get("paragraph_length", "기타")] += 1

            per_problem.append(analysis)

    summary = {
        "total_test_questions": total,
        "korean_history_questions": korean_history_count,
        "korean_history_ratio": round(korean_history_count / total, 3),
        "problem_type_distribution": dict(type_counter),
        "marker_distribution": dict(marker_counter),
        "paragraph_length_distribution": dict(length_counter)
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_problem": per_problem
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("✅ 한국사 test 구조 분석 완료")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
