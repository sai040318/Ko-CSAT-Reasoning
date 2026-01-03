# /data/ephemeral/home/.local/bin/uv run
"""
CSV 파일 비교 분석 스크립트
원본 train.csv와 data_1_1_0/train_all.csv의 차이점 분석
train_all.csv의 정보를 분석하기 위해 생성했던 스크립트
"""

import argparse
import pandas as pd


def compare_csv_files(original_path: str, combined_path: str) -> None:
    """
    두 CSV 파일을 비교 분석

    Args:
        original_path: 원본 CSV 파일 경로
        combined_path: 비교할 CSV 파일 경로
    """
    # CSV 파일 읽기
    original = pd.read_csv(original_path, encoding="utf-8-sig")
    train_all = pd.read_csv(combined_path, encoding="utf-8-sig")

    print("=" * 60)
    print("CSV 파일 비교 분석")
    print("=" * 60)

    # 기본 정보
    print(f"\n[1] 기본 정보")
    print(f"  원본 ({original_path}): {len(original)}개 데이터")
    print(f"  비교 대상 ({combined_path}): {len(train_all)}개 데이터")
    print(f"  차이: {len(original) - len(train_all)}개")

    # 컬럼 비교
    print(f"\n[2] 컬럼 비교")
    print(f"  원본 컬럼: {list(original.columns)}")
    print(f"  비교 대상 컬럼: {list(train_all.columns)}")

    # ID 비교
    print(f"\n[3] ID 비교")
    original_ids = set(original["id"].tolist())
    combined_ids = set(train_all["id"].tolist())

    only_in_original = original_ids - combined_ids
    only_in_combined = combined_ids - original_ids
    common_ids = original_ids & combined_ids

    print(f"  공통 ID 개수: {len(common_ids)}개")
    print(f"  원본에만 있는 ID 개수: {len(only_in_original)}개")
    print(f"  비교 대상에만 있는 ID 개수: {len(only_in_combined)}개")

    if only_in_original:
        print(f"\n[4] 원본에만 있는 ID 목록 (처음 20개):")
        sorted_only_original = sorted(
            only_in_original, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
        )
        for i, id_val in enumerate(sorted_only_original[:20]):
            print(f"    {i + 1}. {id_val}")
        if len(only_in_original) > 20:
            print(f"    ... 외 {len(only_in_original) - 20}개")

    if only_in_combined:
        print(f"\n[5] 비교 대상에만 있는 ID 목록 (처음 20개):")
        sorted_only_combined = sorted(
            only_in_combined, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
        )
        for i, id_val in enumerate(sorted_only_combined[:20]):
            print(f"    {i + 1}. {id_val}")
        if len(only_in_combined) > 20:
            print(f"    ... 외 {len(only_in_combined) - 20}개")

    # 중복 ID 확인
    print(f"\n[6] 중복 ID 확인")
    original_duplicates = original[original["id"].duplicated(keep=False)]
    combined_duplicates = train_all[train_all["id"].duplicated(keep=False)]

    print(f"  원본 중복 ID 개수: {len(original_duplicates)}개 (고유 ID: {original_duplicates['id'].nunique()}개)")
    print(f"  비교 대상 중복 ID 개수: {len(combined_duplicates)}개 (고유 ID: {combined_duplicates['id'].nunique()}개)")

    if len(original_duplicates) > 0:
        print(f"\n  원본 중복 ID 목록:")
        for id_val in original_duplicates["id"].unique()[:10]:
            count = len(original[original["id"] == id_val])
            print(f"    - {id_val}: {count}번 중복")

    if len(combined_duplicates) > 0:
        print(f"\n  비교 대상 중복 ID 목록:")
        for id_val in combined_duplicates["id"].unique()[:10]:
            count = len(train_all[train_all["id"] == id_val])
            print(f"    - {id_val}: {count}번 중복")

    # 공통 ID의 내용 차이 확인
    print(f"\n[7] 공통 ID의 내용 차이 확인")
    diff_count = 0
    diff_examples = []

    for id_val in list(common_ids)[:100]:  # 처음 100개만 확인
        orig_row = original[original["id"] == id_val].iloc[0]
        comb_row = train_all[train_all["id"] == id_val].iloc[0]

        for col in original.columns:
            if col in train_all.columns:
                orig_val = str(orig_row[col])
                comb_val = str(comb_row[col])
                if orig_val != comb_val:
                    diff_count += 1
                    if len(diff_examples) < 5:
                        diff_examples.append(
                            {
                                "id": id_val,
                                "column": col,
                                "original": orig_val[:100] + "..." if len(orig_val) > 100 else orig_val,
                                "train_all": comb_val[:100] + "..." if len(comb_val) > 100 else comb_val,
                            }
                        )

    print(f"  내용이 다른 경우: {diff_count}건 (처음 100개 ID 중)")

    if diff_examples:
        print(f"\n  내용 차이 예시:")
        for i, diff in enumerate(diff_examples):
            print(f"\n    예시 {i + 1}: ID = {diff['id']}, 컬럼 = {diff['column']}")
            print(f"      원본: {diff['original']}")
            print(f"      비교: {diff['train_all']}")

    # ID 범위 분석
    print(f"\n[8] ID 숫자 범위 분석")

    def extract_num(id_val):
        parts = id_val.split("-")
        if parts[-1].isdigit():
            return int(parts[-1])
        return None

    original_nums = [extract_num(x) for x in original_ids if extract_num(x) is not None]
    combined_nums = [extract_num(x) for x in combined_ids if extract_num(x) is not None]

    if original_nums:
        print(f"  원본 ID 범위: {min(original_nums)} ~ {max(original_nums)}")
    if combined_nums:
        print(f"  비교 대상 ID 범위: {min(combined_nums)} ~ {max(combined_nums)}")

    # 누락된 ID 범위 분석
    if only_in_original:
        missing_nums = sorted([extract_num(x) for x in only_in_original if extract_num(x) is not None])
        if missing_nums:
            print(f"\n[9] 누락된 ID 상세 분석")
            print(f"  누락된 ID 숫자 범위: {min(missing_nums)} ~ {max(missing_nums)}")

            # 연속된 범위 찾기
            ranges = []
            start = missing_nums[0]
            end = missing_nums[0]
            for num in missing_nums[1:]:
                if num == end + 1:
                    end = num
                else:
                    ranges.append((start, end))
                    start = num
                    end = num
            ranges.append((start, end))

            print(f"  누락된 ID 연속 범위:")
            for start, end in ranges[:10]:
                if start == end:
                    print(f"    - {start}")
                else:
                    print(f"    - {start} ~ {end} ({end - start + 1}개)")
            if len(ranges) > 10:
                print(f"    ... 외 {len(ranges) - 10}개 범위")


def main():
    parser = argparse.ArgumentParser(description="CSV 파일 비교 분석")
    parser.add_argument("--original", required=True, help="원본 CSV 파일 경로")
    parser.add_argument("--compare", required=True, help="비교할 CSV 파일 경로")

    args = parser.parse_args()

    compare_csv_files(
        original_path=args.original,
        combined_path=args.compare,
    )


if __name__ == "__main__":
    main()
