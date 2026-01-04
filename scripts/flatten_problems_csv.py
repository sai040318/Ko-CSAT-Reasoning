#!/usr/bin/env python3
"""
CSV 파일의 problems JSON 컬럼을 펼쳐서 보기 좋은 형태로 변환하는 스크립트.

Usage:
    python scripts/flatten_problems_csv.py --input data/train.csv --output data/train_flattened.csv
    python scripts/flatten_problems_csv.py --input data/test.csv --output data/test_flattened.csv
    python scripts/flatten_problems_csv.py --input data/train.csv  # 기본 출력: data/train_flattened.csv
"""

import argparse
import pandas as pd
from ast import literal_eval
from pathlib import Path


def flatten_problems(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    CSV 파일의 problems JSON 컬럼을 펼쳐서 별도 컬럼으로 분리.

    Args:
        input_path: 입력 CSV 파일 경로
        output_path: 출력 CSV 파일 경로 (None이면 자동 생성)

    Returns:
        펼쳐진 DataFrame
    """
    # CSV 로드
    df = pd.read_csv(input_path)
    print(f"원본 파일 로드: {input_path}")
    print(f"원본 컬럼: {list(df.columns)}")
    print(f"총 행 수: {len(df)}")

    # 인덱스 컬럼이 있으면 제거 (Unnamed: 0 같은)
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"제거된 인덱스 컬럼: {unnamed_cols}")

    records = []
    for idx, row in df.iterrows():
        try:
            problems = literal_eval(row["problems"])
        except (ValueError, SyntaxError) as e:
            print(f"경고: {idx}번 행의 problems 파싱 실패 - {e}")
            problems = {}

        # 기본 정보
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
        }

        # problems에서 추출
        record["question"] = problems.get("question", "")
        record["answer"] = problems.get("answer", "")

        # choices를 개별 컬럼으로 펼치기
        choices = problems.get("choices", [])
        for i, choice in enumerate(choices, start=1):
            record[f"choice_{i}"] = choice

        # 빈 choice 컬럼 채우기 (최대 5개 선택지 가정)
        for i in range(1, 6):
            if f"choice_{i}" not in record:
                record[f"choice_{i}"] = ""

        # question_plus 처리 (row 또는 problems 안에 있을 수 있음)
        question_plus = row.get("question_plus", None)
        if pd.isna(question_plus) or question_plus is None:
            question_plus = problems.get("question_plus", "")
        record["question_plus"] = question_plus if not pd.isna(question_plus) else ""

        records.append(record)

    # DataFrame 생성
    flattened_df = pd.DataFrame(records)

    # 컬럼 순서 정리 (question_plus를 선지 앞에 배치)
    column_order = [
        "id", "paragraph", "question", "question_plus",
        "choice_1", "choice_2", "choice_3", "choice_4", "choice_5",
        "answer"
    ]
    # 실제 존재하는 컬럼만 선택
    column_order = [col for col in column_order if col in flattened_df.columns]
    flattened_df = flattened_df[column_order]

    # 빈 컬럼 제거 (모든 값이 빈 문자열인 choice 컬럼)
    for col in list(flattened_df.columns):
        if col.startswith("choice_"):
            if flattened_df[col].replace("", pd.NA).isna().all():
                flattened_df = flattened_df.drop(columns=[col])

    # 출력 경로 설정 (data/flattened 디렉토리에 저장)
    if output_path is None:
        input_path_obj = Path(input_path)
        flattened_dir = input_path_obj.parent / "flattened"
        flattened_dir.mkdir(exist_ok=True)
        output_path = flattened_dir / f"{input_path_obj.stem}_flattened.csv"

    # CSV 저장
    flattened_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n변환 완료!")
    print(f"출력 파일: {output_path}")
    print(f"새 컬럼: {list(flattened_df.columns)}")

    # 샘플 출력
    print(f"\n=== 첫 번째 행 샘플 ===")
    first_row = flattened_df.iloc[0]
    for col in flattened_df.columns:
        value = str(first_row[col])
        if len(value) > 100:
            value = value[:100] + "..."
        print(f"  {col}: {value}")

    return flattened_df


def main():
    parser = argparse.ArgumentParser(
        description="CSV의 problems JSON 컬럼을 펼쳐서 보기 좋은 형태로 변환"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 CSV 파일 경로"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 CSV 파일 경로 (기본: 입력파일명_flattened.csv)"
    )

    args = parser.parse_args()
    flatten_problems(args.input, args.output)


if __name__ == "__main__":
    main()
