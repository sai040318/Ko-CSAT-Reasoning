# /data/ephemeral/home/.local/bin/uv run
"""
CSV 파일 합치기 스크립트
train.csv와 eval.csv를 합치고 id 순으로 정렬
"""

import argparse
import pandas as pd
from pathlib import Path


def concat_and_sort_csv(
    train_path: str,
    eval_path: str,
    output_path: str,
    sort_column: str = "id",
) -> pd.DataFrame:
    """
    두 CSV 파일을 합치고 정렬하여 저장

    Args:
        train_path: train.csv 경로
        eval_path: eval.csv 경로
        output_path: 출력 파일 경로
        sort_column: 정렬 기준 컬럼 (기본값: id)

    Returns:
        합쳐진 DataFrame
    """
    # CSV 파일 읽기
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    eval_df = pd.read_csv(eval_path, encoding="utf-8-sig")

    print(f"train.csv: {len(train_df)}개 데이터")
    print(f"eval.csv: {len(eval_df)}개 데이터")

    # 합치기
    combined = pd.concat([train_df, eval_df], ignore_index=True)
    print(f"합친 후: {len(combined)}개 데이터")

    # id에서 숫자 추출하여 정렬 (예: generation-for-nlp-123 -> 123)
    if sort_column == "id" and combined[sort_column].str.contains("-").any():
        combined["_sort_key"] = combined[sort_column].str.extract(r"(\d+)$").astype(int)
        combined = combined.sort_values("_sort_key").drop("_sort_key", axis=1)
    else:
        combined = combined.sort_values(sort_column)

    combined = combined.reset_index(drop=True)

    # 저장
    combined.to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="CSV 파일 합치기")
    parser.add_argument("--train", required=True, help="train.csv 경로")
    parser.add_argument("--eval", required=True, help="eval.csv 경로")
    parser.add_argument("--output", required=True, help="출력 파일 경로")
    parser.add_argument("--sort-column", default="id", help="정렬 기준 컬럼 (기본값: id)")

    args = parser.parse_args()

    concat_and_sort_csv(
        train_path=args.train,
        eval_path=args.eval,
        output_path=args.output,
        sort_column=args.sort_column,
    )


if __name__ == "__main__":
    main()
