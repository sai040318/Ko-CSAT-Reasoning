# make_constant_submissions.py
# Usage:
#   python make_constant_submissions.py --input select_all_1.csv --out_dir .
#
# Output:
#   submission_all1.csv ~ submission_all5.csv
#   (각 파일은 answer 컬럼을 1~5로 전부 동일하게 채움)

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="select_all_1.csv", help="기준 CSV 파일 경로")
    parser.add_argument("--out_dir", type=str, default=".", help="출력 디렉터리")
    parser.add_argument("--answer_col", type=str, default="answer", help="정답(선지번호) 컬럼명")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.answer_col not in df.columns:
        raise ValueError(
            f"'{args.answer_col}' 컬럼을 찾을 수 없습니다. "
            f"현재 컬럼: {list(df.columns)} / --answer_col 로 컬럼명을 지정하세요."
        )

    # 1~5로 각각 모두 찍은 submission 생성
    for choice in range(1, 6):
        out_df = df.copy()
        out_df[args.answer_col] = choice

        out_path = out_dir / f"submission_all{choice}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Done. Generated 5 files in: {out_dir.resolve()}")
    for choice in range(1, 6):
        print(f" - {out_dir / f'submission_all{choice}.csv'}")


if __name__ == "__main__":
    main()
