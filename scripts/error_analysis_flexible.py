"""
에러 케이스 분석 스크립트 (유연한 버전)

사용법:
    python scripts/error_analysis_flexible.py --data <데이터파일> --pred <예측파일>

예시:
    python scripts/error_analysis_flexible.py \
        --data tmp/data_augmented_1_0_0_flatten_augmented.csv \
        --pred tmp/qwen3_2507_ollama_instruct_0104_151720_output.csv
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser(description="에러 케이스 분석 스크립트")
    parser.add_argument("--data", required=True, help="정답이 포함된 데이터 파일 경로")
    parser.add_argument("--pred", required=True, help="모델 예측 결과 파일 경로")
    parser.add_argument("--output-dir", default=".", help="결과 저장 디렉토리 (기본: 현재 디렉토리)")
    parser.add_argument("--id-col", default="id", help="ID 컬럼명 (기본: id)")
    parser.add_argument("--answer-col", default="answer", help="정답 컬럼명 (기본: answer)")
    parser.add_argument("--pred-col", default="answer", help="예측 컬럼명 (기본: answer)")
    return parser.parse_args()


def load_data(data_path: str, pred_path: str, id_col: str, answer_col: str, pred_col: str):
    """데이터 로드 및 병합"""
    data_df = pd.read_csv(data_path)
    pred_df = pd.read_csv(pred_path)

    # 예측 컬럼명이 정답 컬럼명과 같으면 구분을 위해 이름 변경
    pred_df = pred_df.rename(columns={pred_col: "predicted"})

    # ID 기준 병합
    merged = data_df.merge(pred_df[[id_col, "predicted"]], on=id_col, how="inner")
    return merged, data_df, pred_df


def calculate_metrics(y_true, y_pred, labels=None):
    """성능 지표 계산"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {
            "labels": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
    }


def print_confusion_matrix(y_true, y_pred, labels=None):
    """Confusion Matrix 출력"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n=== Confusion Matrix ===")
    print("(행: 실제, 열: 예측)")

    # 헤더
    header = "       " + "  ".join(f"Pred_{l:>2}" for l in labels)
    print(header)

    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"True_{labels[i]:>2} {row_str}")

    return cm


def analyze_errors(merged_df, answer_col: str):
    """에러 케이스 분석"""
    errors = merged_df[merged_df[answer_col] != merged_df["predicted"]].copy()
    correct = merged_df[merged_df[answer_col] == merged_df["predicted"]].copy()

    # 에러 케이스에 ground_truth 컬럼 추가
    errors = errors.rename(columns={answer_col: "ground_truth"})

    return errors, correct


def analyze_error_patterns(errors_df):
    """에러 패턴 분석"""
    if errors_df.empty:
        return None

    # 예측값별 에러 분포
    pred_dist = errors_df["predicted"].value_counts().sort_index()

    # 실제값별 에러 분포
    true_dist = errors_df["ground_truth"].value_counts().sort_index()

    # 예측 -> 실제 전환 패턴
    transition = (
        errors_df.groupby(["predicted", "ground_truth"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return {
        "pred_distribution": pred_dist,
        "true_distribution": true_dist,
        "transition_patterns": transition,
    }


def print_distribution(series, name: str, total: int):
    """분포 출력"""
    print(f"\n    {name}:")
    for label, count in series.items():
        pct = count / total * 100
        print(f"        클래스 {label}: {count}개 ({pct:.1f}%)")


def main():
    args = parse_args()

    print("=" * 60)
    print("Error Case Analysis & Scoring")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    merged, data_df, pred_df = load_data(
        args.data, args.pred, args.id_col, args.answer_col, args.pred_col
    )
    print(f"    - 데이터 파일: {args.data} ({len(data_df)}개)")
    print(f"    - 예측 파일: {args.pred} ({len(pred_df)}개)")
    print(f"    - 매칭된 샘플: {len(merged)}개")

    # 2. 클래스 분포
    print("\n[2] 클래스 분포")
    y_true = merged[args.answer_col].tolist()
    y_pred = merged["predicted"].tolist()
    labels = sorted(set(y_true) | set(y_pred))

    print_distribution(pd.Series(y_true).value_counts().sort_index(), "Ground Truth 분포", len(y_true))
    print_distribution(pd.Series(y_pred).value_counts().sort_index(), "예측값 분포", len(y_pred))

    # 3. Scoring
    print("\n" + "=" * 60)
    print("[3] SCORING RESULTS")
    print("=" * 60)

    metrics = calculate_metrics(y_true, y_pred, labels)
    correct_count = sum(t == p for t, p in zip(y_true, y_pred))

    print(f"\n    Accuracy: {metrics['accuracy']:.4f} ({correct_count}/{len(y_true)})")
    print(f"    Macro F1-score: {metrics['macro_f1']:.4f}")

    # 클래스별 성능
    print("\n    클래스별 성능:")
    print("    " + "-" * 50)
    print(f"    {'클래스':^6} {'Precision':^10} {'Recall':^10} {'F1':^10} {'Support':^8}")
    print("    " + "-" * 50)

    pc = metrics["per_class"]
    for i, label in enumerate(pc["labels"]):
        print(
            f"    {label:^6} {pc['precision'][i]:^10.4f} {pc['recall'][i]:^10.4f} "
            f"{pc['f1'][i]:^10.4f} {int(pc['support'][i]):^8}"
        )
    print("    " + "-" * 50)

    # Confusion Matrix
    print_confusion_matrix(y_true, y_pred, labels)

    # 4. Error Case Analysis
    print("\n" + "=" * 60)
    print("[4] ERROR CASE ANALYSIS")
    print("=" * 60)

    errors_df, correct_df = analyze_errors(merged, args.answer_col)
    print(f"\n    총 에러 수: {len(errors_df)}")
    print(f"    정답 수: {len(correct_df)}")

    # 에러 패턴 분석
    patterns = analyze_error_patterns(errors_df)
    if patterns:
        print("\n    [에러 패턴 분석]")
        print("\n    예측값별 에러 분포 (모델이 잘못 예측한 답):")
        for label, count in patterns["pred_distribution"].items():
            print(f"        클래스 {label}: {count}개")

        print("\n    실제값별 에러 분포 (모델이 틀린 문제의 정답):")
        for label, count in patterns["true_distribution"].items():
            print(f"        클래스 {label}: {count}개")

        print("\n    주요 오분류 패턴 (Top 10):")
        print("    " + "-" * 40)
        print(f"    {'예측':^6} {'실제':^6} {'빈도':^8}")
        print("    " + "-" * 40)
        for _, row in patterns["transition_patterns"].head(10).iterrows():
            print(f"    {row['predicted']:^6} {row['ground_truth']:^6} {int(row['count']):^8}")

    # 5. 에러 케이스 샘플
    print("\n" + "=" * 60)
    print("[5] ERROR CASE SAMPLES (처음 10개)")
    print("=" * 60)

    # 표시할 컬럼 결정 (있는 컬럼만)
    display_cols = ["id", "predicted", "ground_truth"]
    optional_cols = ["paragraph", "question", "question_plus", "choice_1", "choice_2", "choice_3", "choice_4", "choice_5"]
    for col in optional_cols:
        if col in errors_df.columns:
            display_cols.append(col)

    for i, (_, row) in enumerate(errors_df.head(10).iterrows()):
        print(f"\n--- Error Case #{i + 1} ---")
        print(f"ID: {row.get(args.id_col, 'N/A')}")
        print(f"예측: {row['predicted']} / 정답: {row['ground_truth']}")

        if "question" in row:
            print(f"질문: {row['question']}")
        if "paragraph" in row:
            para = str(row["paragraph"])
            print(f"지문: {para[:200]}..." if len(para) > 200 else f"지문: {para}")

        # 선택지 출력
        choices = []
        for j in range(1, 6):
            col = f"choice_{j}"
            if col in row and pd.notna(row[col]):
                choices.append(f"{j}. {row[col]}")
        if choices:
            print("선택지:")
            for c in choices:
                print(f"    {c}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("[6] SUMMARY")
    print("=" * 60)
    print(f"""
    - 총 평가 샘플: {len(merged)}개
    - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)
    - Macro F1-score: {metrics['macro_f1']:.4f}
    - 에러 케이스: {len(errors_df)}개
    - 정답 케이스: {len(correct_df)}개

    [클래스별 F1-score]""")
    for i, label in enumerate(pc["labels"]):
        print(f"    - 클래스 {label}: {pc['f1'][i]:.4f}")

    # 7. 에러 케이스 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    error_output_path = output_dir / f"error_cases_{timestamp}.csv"
    errors_df.to_csv(error_output_path, index=False)
    print(f"\n에러 케이스가 저장되었습니다: {error_output_path}")


if __name__ == "__main__":
    main()
