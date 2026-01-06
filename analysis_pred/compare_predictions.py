#!/usr/bin/env python3
"""
CSV 파일 예측 비교 분석 스크립트
두 개의 CSV 파일에서 서로 다르게 예측한 항목들을 상세하게 분석합니다.
"""

import pandas as pd
import sys
from pathlib import Path


def load_csv_files(file1_path, file2_path):
    """CSV 파일들을 로드합니다."""
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        return df1, df2
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        sys.exit(1)


def compare_predictions(df1, df2, name1, name2):
    """두 CSV 파일의 예측을 비교하고 상세 분석을 출력합니다."""

    # 파일명 표시
    print("=" * 100)
    print(f"예측 비교 분석 리포트")
    print("=" * 100)
    print(f"\n파일 1: {name1}")
    print(f"파일 2: {name2}\n")

    # 기본 정보
    print("-" * 100)
    print("1. 기본 정보")
    print("-" * 100)
    print(f"파일 1 총 레코드 수: {len(df1)}")
    print(f"파일 2 총 레코드 수: {len(df2)}")

    # 두 파일을 id 기준으로 병합
    merged = pd.merge(df1, df2, on="id", how="outer", suffixes=("_file1", "_file2"))

    # 양쪽에 모두 존재하는 레코드
    both_exist = merged[merged["answer_file1"].notna() & merged["answer_file2"].notna()]
    print(f"양쪽 파일에 모두 존재하는 레코드 수: {len(both_exist)}")

    # 한쪽에만 존재하는 레코드
    only_file1 = merged[merged["answer_file1"].notna() & merged["answer_file2"].isna()]
    only_file2 = merged[merged["answer_file1"].isna() & merged["answer_file2"].notna()]

    if len(only_file1) > 0:
        print(f"\n파일 1에만 존재하는 레코드 수: {len(only_file1)}")
        print(f"ID 목록: {only_file1['id'].tolist()}")

    if len(only_file2) > 0:
        print(f"\n파일 2에만 존재하는 레코드 수: {len(only_file2)}")
        print(f"ID 목록: {only_file2['id'].tolist()}")

    # 예측 비교 (양쪽에 모두 존재하는 경우만)
    print("\n" + "-" * 100)
    print("2. 예측 일치/불일치 분석")
    print("-" * 100)

    # 일치하는 경우
    matching = both_exist[both_exist["answer_file1"] == both_exist["answer_file2"]]
    print(f"예측이 일치하는 레코드 수: {len(matching)} ({len(matching) / len(both_exist) * 100:.2f}%)")

    # 불일치하는 경우
    different = both_exist[both_exist["answer_file1"] != both_exist["answer_file2"]]
    print(f"예측이 다른 레코드 수: {len(different)} ({len(different) / len(both_exist) * 100:.2f}%)")

    # 불일치 상세 분석
    if len(different) > 0:
        print("\n" + "-" * 100)
        print("3. 예측이 다른 레코드 상세 분석")
        print("-" * 100)

        # 정렬하여 출력
        different_sorted = different.sort_values("id")

        print(f"\n{'ID':<30} | {'파일1 예측':<15} | {'파일2 예측':<15}")
        print("-" * 100)

        for idx, row in different_sorted.iterrows():
            print(f"{row['id']:<30} | {str(row['answer_file1']):<15} | {str(row['answer_file2']):<15}")

        # 예측값 분포 분석
        print("\n" + "-" * 100)
        print("4. 불일치 케이스의 예측값 분포")
        print("-" * 100)

        print("\n파일 1의 예측값 분포 (불일치 케이스):")
        print(different["answer_file1"].value_counts().sort_index())

        print("\n파일 2의 예측값 분포 (불일치 케이스):")
        print(different["answer_file2"].value_counts().sort_index())

        # 예측 변화 패턴 분석
        print("\n" + "-" * 100)
        print("5. 예측 변화 패턴 분석")
        print("-" * 100)

        change_pattern = different.groupby(["answer_file1", "answer_file2"]).size().reset_index(name="count")
        change_pattern = change_pattern.sort_values("count", ascending=False)

        print(f"\n{'파일1 → 파일2':<30} | {'발생 횟수':<15}")
        print("-" * 100)

        for idx, row in change_pattern.iterrows():
            print(f"{str(row['answer_file1'])} → {str(row['answer_file2']):<20} | {row['count']:<15}")

    # 전체 예측값 분포
    print("\n" + "-" * 100)
    print("6. 전체 예측값 분포")
    print("-" * 100)

    print("\n파일 1의 전체 예측값 분포:")
    print(df1["answer"].value_counts().sort_index())

    print("\n파일 2의 전체 예측값 분포:")
    print(df2["answer"].value_counts().sort_index())

    print("\n" + "=" * 100)
    print("분석 완료")
    print("=" * 100)

    return different


def main():
    # 파일 경로 설정
    script_dir = Path(__file__).parent
    file1_path = script_dir / "qwen3_2507_ollama_thinking_param_0105_163938_output.csv"
    file2_path = script_dir / "qwen3_2507_ollama_thinking_param_0106_004147_output.csv"

    # 파일 존재 여부 확인
    if not file1_path.exists():
        print(f"오류: {file1_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    if not file2_path.exists():
        print(f"오류: {file2_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # CSV 파일 로드
    df1, df2 = load_csv_files(file1_path, file2_path)

    # 비교 분석 실행
    different = compare_predictions(df1, df2, file1_path.name, file2_path.name)

    # 결과를 CSV 파일로 저장
    if len(different) > 0:
        output_path = script_dir / "prediction_differences.csv"
        different_sorted = different.sort_values("id")
        different_sorted.to_csv(output_path, index=False)
        print(f"\n불일치 레코드가 {output_path} 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
