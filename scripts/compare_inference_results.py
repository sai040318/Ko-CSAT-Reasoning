"""
LLM Inference Consistency Analysis Script

두 개의 inference 결과 CSV 파일을 비교하여 LLM의 일관성을 분석합니다.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import Counter, defaultdict


def load_csv_files(file1: Path, file2: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """CSV 파일을 로드하고 기본 검증을 수행합니다."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 기본 검증
    assert "id" in df1.columns and "answer" in df1.columns, f"{file1}에 'id', 'answer' 컬럼이 필요합니다."
    assert "id" in df2.columns and "answer" in df2.columns, f"{file2}에 'id', 'answer' 컬럼이 필요합니다."

    # ID로 정렬
    df1 = df1.sort_values("id").reset_index(drop=True)
    df2 = df2.sort_values("id").reset_index(drop=True)

    return df1, df2


def basic_statistics(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """기본 통계를 계산합니다."""
    merged = pd.merge(df1, df2, on="id", suffixes=("_baseline", "_current"))

    total_samples = len(merged)
    same_predictions = (merged["answer_baseline"] == merged["answer_current"]).sum()
    diff_predictions = total_samples - same_predictions

    consistency_rate = (same_predictions / total_samples * 100) if total_samples > 0 else 0

    return {
        "total_samples": total_samples,
        "same_predictions": same_predictions,
        "different_predictions": diff_predictions,
        "consistency_rate": consistency_rate,
        "baseline_file_samples": len(df1),
        "current_file_samples": len(df2),
    }


def analyze_changes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """변경된 항목들을 상세히 분석합니다."""
    merged = pd.merge(df1, df2, on="id", suffixes=("_baseline", "_current"))

    # 변경된 항목만 필터링
    changed = merged[merged["answer_baseline"] != merged["answer_current"]].copy()

    if len(changed) > 0:
        # 변경 크기 계산
        changed["change_magnitude"] = abs(changed["answer_current"] - changed["answer_baseline"])
        changed["change_direction"] = changed["answer_current"] - changed["answer_baseline"]
        changed["change_type"] = changed.apply(
            lambda row: f"{int(row['answer_baseline'])} → {int(row['answer_current'])}", axis=1
        )

    return changed


def change_matrix(changed_df: pd.DataFrame) -> pd.DataFrame:
    """변경 패턴을 매트릭스 형태로 분석합니다."""
    if len(changed_df) == 0:
        return pd.DataFrame()

    # 변경 매트릭스 생성 (from -> to)
    change_counts = defaultdict(lambda: defaultdict(int))

    for _, row in changed_df.iterrows():
        from_answer = int(row["answer_baseline"])
        to_answer = int(row["answer_current"])
        change_counts[from_answer][to_answer] += 1

    # DataFrame으로 변환
    all_answers = sorted(set(changed_df["answer_baseline"].unique()) | set(changed_df["answer_current"].unique()))
    matrix = pd.DataFrame(0, index=all_answers, columns=all_answers)

    for from_ans, to_dict in change_counts.items():
        for to_ans, count in to_dict.items():
            matrix.loc[from_ans, to_ans] = count

    return matrix


def distribution_analysis(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """답변 분포 변화를 분석합니다."""
    dist1 = df1["answer"].value_counts().sort_index()
    dist2 = df2["answer"].value_counts().sort_index()

    # 모든 답변 옵션 (1-5)을 포함
    all_answers = range(1, 6)
    dist1 = dist1.reindex(all_answers, fill_value=0)
    dist2 = dist2.reindex(all_answers, fill_value=0)

    # 변화량 계산
    distribution_changes = {
        "baseline": dist1.to_dict(),
        "current": dist2.to_dict(),
        "absolute_change": (dist2 - dist1).to_dict(),
        "percentage_change": ((dist2 - dist1) / dist1 * 100).replace([np.inf, -np.inf], 0).fillna(0).to_dict(),
    }

    return distribution_changes


def magnitude_analysis(changed_df: pd.DataFrame) -> Dict:
    """변경 크기별 분석을 수행합니다."""
    if len(changed_df) == 0:
        return {}

    magnitude_counts = changed_df["change_magnitude"].value_counts().sort_index()
    direction_stats = {
        "increased": (changed_df["change_direction"] > 0).sum(),
        "decreased": (changed_df["change_direction"] < 0).sum(),
        "avg_change": changed_df["change_direction"].mean(),
        "avg_absolute_change": changed_df["change_magnitude"].mean(),
    }

    return {
        "magnitude_distribution": magnitude_counts.to_dict(),
        "direction_statistics": direction_stats,
    }


def pattern_analysis(changed_df: pd.DataFrame) -> Dict:
    """ID 패턴별 변경 분석을 수행합니다."""
    if len(changed_df) == 0:
        return {}

    # ID에서 숫자만 추출하여 분포 확인
    changed_df["id_number"] = changed_df["id"].str.extract(r"(\d+)$").astype(int)

    # 변경이 많이 발생한 ID 범위 분석 (100단위로)
    changed_df["id_range"] = (changed_df["id_number"] // 100) * 100
    range_counts = changed_df["id_range"].value_counts().sort_index()

    # 가장 변경이 많이 발생한 답변
    most_changed_from = changed_df["answer_baseline"].mode().values[0] if len(changed_df) > 0 else None
    most_changed_to = changed_df["answer_current"].mode().values[0] if len(changed_df) > 0 else None

    # 변경 타입별 빈도
    change_type_counts = changed_df["change_type"].value_counts().head(10)

    return {
        "id_range_distribution": range_counts.to_dict(),
        "most_changed_from_answer": int(most_changed_from) if most_changed_from is not None else None,
        "most_changed_to_answer": int(most_changed_to) if most_changed_to is not None else None,
        "top_10_change_patterns": change_type_counts.to_dict(),
    }


def stability_score_by_answer(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """각 답변별 안정성 점수를 계산합니다."""
    merged = pd.merge(df1, df2, on="id", suffixes=("_baseline", "_current"))

    stability_scores = {}
    for answer in range(1, 6):
        baseline_count = (merged["answer_baseline"] == answer).sum()
        if baseline_count > 0:
            # baseline에서 해당 답변이었던 것 중 current에서도 같은 답변인 비율
            stable_count = ((merged["answer_baseline"] == answer) & (merged["answer_current"] == answer)).sum()
            stability_scores[answer] = {
                "count": baseline_count,
                "stable_count": stable_count,
                "stability_rate": (stable_count / baseline_count * 100),
            }

    return stability_scores


def print_report(
    stats: Dict,
    changed_df: pd.DataFrame,
    dist_analysis: Dict,
    mag_analysis: Dict,
    pattern_analysis: Dict,
    stability_scores: Dict,
    change_mat: pd.DataFrame,
):
    """분석 결과를 출력합니다."""
    print("=" * 80)
    print("LLM Inference Consistency Analysis Report")
    print("=" * 80)
    print()

    # 1. 기본 통계
    print("📊 1. BASIC STATISTICS")
    print("-" * 80)
    print(f"Total Samples:              {stats['total_samples']:,}")
    print(f"Same Predictions:           {stats['same_predictions']:,} ({stats['consistency_rate']:.2f}%)")
    print(f"Different Predictions:      {stats['different_predictions']:,} ({100 - stats['consistency_rate']:.2f}%)")
    print(f"Baseline File Samples:      {stats['baseline_file_samples']:,}")
    print(f"Current File Samples:       {stats['current_file_samples']:,}")
    print()

    # 2. 변경된 항목 요약
    print("🔄 2. CHANGED ITEMS SUMMARY")
    print("-" * 80)
    if len(changed_df) > 0:
        print(f"Total Changed Items:        {len(changed_df):,}")
        print(f"\nFirst 20 Changed Items:")
        print(
            changed_df[["id", "answer_baseline", "answer_current", "change_type", "change_magnitude"]]
            .head(20)
            .to_string(index=False)
        )
    else:
        print("No changes detected - Perfect consistency! 🎉")
    print()

    # 3. 변경 매트릭스
    if len(change_mat) > 0:
        print("📈 3. CHANGE MATRIX (Baseline → Current)")
        print("-" * 80)
        print("Rows: Baseline answers, Columns: Current answers")
        print(change_mat.to_string())
        print()

    # 4. 분포 변화
    print("📊 4. ANSWER DISTRIBUTION CHANGES")
    print("-" * 80)
    print(f"{'Answer':<10} {'Baseline':<12} {'Current':<12} {'Change':<12} {'% Change':<12}")
    print("-" * 80)
    for answer in range(1, 6):
        baseline_count = dist_analysis["baseline"].get(answer, 0)
        current_count = dist_analysis["current"].get(answer, 0)
        abs_change = dist_analysis["absolute_change"].get(answer, 0)
        pct_change = dist_analysis["percentage_change"].get(answer, 0)
        print(f"{answer:<10} {baseline_count:<12} {current_count:<12} {abs_change:<12} {pct_change:>10.2f}%")
    print()

    # 5. 변경 크기 분석
    if mag_analysis:
        print("📏 5. CHANGE MAGNITUDE ANALYSIS")
        print("-" * 80)
        print(f"Increased Predictions:      {mag_analysis['direction_statistics']['increased']:,}")
        print(f"Decreased Predictions:      {mag_analysis['direction_statistics']['decreased']:,}")
        print(f"Average Change:             {mag_analysis['direction_statistics']['avg_change']:+.3f}")
        print(f"Average Absolute Change:    {mag_analysis['direction_statistics']['avg_absolute_change']:.3f}")
        print(f"\nMagnitude Distribution:")
        for magnitude, count in sorted(mag_analysis["magnitude_distribution"].items()):
            print(f"  Change by {int(magnitude)}: {count:,} occurrences")
        print()

    # 6. 패턴 분석
    if pattern_analysis:
        print("🔍 6. PATTERN ANALYSIS")
        print("-" * 80)
        print(f"Most Changed From Answer:   {pattern_analysis.get('most_changed_from_answer', 'N/A')}")
        print(f"Most Changed To Answer:     {pattern_analysis.get('most_changed_to_answer', 'N/A')}")
        print(f"\nTop 10 Change Patterns:")
        for pattern, count in pattern_analysis["top_10_change_patterns"].items():
            print(f"  {pattern}: {count:,} occurrences")
        print(f"\nID Range Distribution (changes per 100 IDs):")
        for id_range, count in sorted(pattern_analysis["id_range_distribution"].items())[:10]:
            print(f"  {id_range}-{id_range + 99}: {count:,} changes")
        print()

    # 7. 답변별 안정성 점수
    print("🎯 7. STABILITY SCORE BY ANSWER")
    print("-" * 80)
    print(f"{'Answer':<10} {'Count':<12} {'Stable':<12} {'Stability Rate':<15}")
    print("-" * 80)
    for answer in range(1, 6):
        if answer in stability_scores:
            score = stability_scores[answer]
            print(f"{answer:<10} {score['count']:<12} {score['stable_count']:<12} {score['stability_rate']:>13.2f}%")
    print()

    # 8. 인사이트 및 권장사항
    print("💡 8. INSIGHTS & RECOMMENDATIONS")
    print("-" * 80)

    if stats["consistency_rate"] >= 95:
        print("✅ EXCELLENT: Consistency rate is very high (≥95%).")
        print("   The model shows strong stability across runs.")
    elif stats["consistency_rate"] >= 90:
        print("✅ GOOD: Consistency rate is high (≥90%).")
        print("   The model is fairly stable with minor variations.")
    elif stats["consistency_rate"] >= 80:
        print("⚠️  MODERATE: Consistency rate is acceptable (≥80%).")
        print("   Consider investigating factors causing variations.")
    else:
        print("❌ LOW: Consistency rate is concerning (<80%).")
        print("   Significant investigation needed into model stability.")

    print()

    # 추가 인사이트
    if mag_analysis:
        avg_change = mag_analysis["direction_statistics"]["avg_change"]
        if abs(avg_change) > 0.5:
            direction = "higher" if avg_change > 0 else "lower"
            print(f"⚠️  The model shows a systematic bias toward {direction} predictions.")

    if pattern_analysis and pattern_analysis.get("most_changed_from_answer") == pattern_analysis.get(
        "most_changed_to_answer"
    ):
        print("⚠️  Circular pattern detected: some answers are being changed back and forth.")

    # 가장 불안정한 답변
    if stability_scores:
        least_stable = min(stability_scores.items(), key=lambda x: x[1]["stability_rate"])
        print(
            f"\n⚠️  Answer {least_stable[0]} is the least stable ({least_stable[1]['stability_rate']:.2f}% stability)."
        )
        print(f"   Consider reviewing predictions with this answer class.")

    print()
    print("=" * 80)
    print("End of Report")
    print("=" * 80)


def save_detailed_changes(changed_df: pd.DataFrame, output_path: Path):
    """변경된 항목의 상세 정보를 CSV로 저장합니다."""
    if len(changed_df) > 0:
        changed_df.to_csv(output_path, index=False)
        print(f"\n💾 Detailed changes saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two LLM inference result CSV files to analyze consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s baseline.csv current.csv
  %(prog)s outputs/run1.csv outputs/run2.csv --output changes.csv
        """,
    )

    parser.add_argument("baseline", type=Path, help="Baseline CSV file (earlier run)")
    parser.add_argument("current", type=Path, help="Current CSV file (later run)")
    parser.add_argument("--output", "-o", type=Path, help="Output CSV file for detailed changes (optional)")

    args = parser.parse_args()

    # 파일 존재 확인
    if not args.baseline.exists():
        print(f"Error: Baseline file not found: {args.baseline}")
        return 1
    if not args.current.exists():
        print(f"Error: Current file not found: {args.current}")
        return 1

    print(f"Loading files...")
    print(f"  Baseline: {args.baseline}")
    print(f"  Current:  {args.current}")
    if args.output:
        print(f"  Output for detailed changes: {args.output}")
    print()

    # 데이터 로드
    df1, df2 = load_csv_files(args.baseline, args.current)

    # 분석 수행
    print("Performing analysis...\n")
    stats = basic_statistics(df1, df2)
    changed_df = analyze_changes(df1, df2)
    change_mat = change_matrix(changed_df)
    dist_analysis = distribution_analysis(df1, df2)
    mag_analysis = magnitude_analysis(changed_df)
    pattern_analysis_result = pattern_analysis(changed_df)
    stability_scores = stability_score_by_answer(df1, df2)

    # 리포트 출력
    print_report(stats, changed_df, dist_analysis, mag_analysis, pattern_analysis_result, stability_scores, change_mat)

    # 상세 변경사항 저장
    if len(changed_df) == 0:
        print("No changes detected - nothing to save.")
    if args.output and len(changed_df) > 0:
        save_detailed_changes(changed_df, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
