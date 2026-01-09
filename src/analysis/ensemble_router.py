"""
간단 앙상블 스크립트.

입력
 - classify: GPT-4o 분류 결과 CSV (id, label[A/B])
 - pred1: RAG Top3 모델 예측 CSV (id, answer)
 - pred2: RAG Top5 모델 예측 CSV (id, answer)
 - pred3: SOTA 모델 예측 CSV (id, answer)

룰 (id 기준 매칭)
 1) label == "B" (비한국사) -> pred3 사용
 2) label == "A" (한국사)
    - pred1 == pred2 -> pred1 사용
    - pred1 != pred2 -> pred3 사용

출력
 - id, answer 를 가진 CSV 저장
"""

import argparse
import pandas as pd


def load_predictions(path, id_col="id", answer_col="answer"):
    df = pd.read_csv(path)
    df = df[[id_col, answer_col]].rename(columns={id_col: "id", answer_col: "answer"})
    return df.set_index("id")["answer"]


def main(args):
    # 분류 결과
    df_cls = pd.read_csv(args.classify)
    label_map = df_cls.set_index("id")["label"]

    # 예측들
    pred1 = load_predictions(args.pred1)
    pred2 = load_predictions(args.pred2)
    pred3 = load_predictions(args.pred3)

    # 공통 id 교집합만 사용 (id 기준 매칭)
    ids = sorted(set(label_map.index) & set(pred1.index) & set(pred2.index) & set(pred3.index))
    results = []

    for _id in ids:
        label = str(label_map[_id]).strip().upper()
        a1 = pred1[_id]
        a2 = pred2[_id]
        a3 = pred3[_id]

        if label == "A":  # 한국사
            if a1 == a2:
                final = a1
            else:
                final = a3  # RAG가 다르면 SOTA 선택
        else:  # 비한국사 혹은 분류 실패 시에도 SOTA 사용
            final = a3

        results.append({"id": _id, "answer": final})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"✅ saved: {args.output} (rows={len(out_df)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify", default="gpt4_classification_results.csv")
    ap.add_argument("--pred1", default="pred_top3.csv")
    ap.add_argument("--pred2", default="pred_top5.csv")
    ap.add_argument("--pred3", default="pred_sota.csv")
    ap.add_argument("--output", default="ensemble_final.csv")
    args = ap.parse_args()
    main(args)
