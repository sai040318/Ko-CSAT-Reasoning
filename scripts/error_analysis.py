"""
Train 데이터에 대한 모델 추론 결과 Error Case Analysis 및 Scoring

평가 지표: Macro F1-score (클래스 1-5에 대해 각각 F1 계산 후 평균)
"""

import pandas as pd
import numpy as np
import date
from ast import literal_eval
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# 파일 경로
OUTPUT_PATH = (
    "/data/ephemeral/home/kdh/outputs/qwen3_2507_thinking_train_0102_221623/qwen3_2507_thinking_0102_221623_output.csv"
)
INPUT_DATA_PATH = "/data/ephemeral/home/kdh/outputs/qwen3_2507_thinking_train_0102_221623/qwen3_2507_thinking_0102_221623_output_input_data.csv"
TRAIN_PATH = "/data/ephemeral/home/kdh/data/train.csv"


def load_data():
    """데이터 로드"""
    predictions_df = pd.read_csv(OUTPUT_PATH)
    input_data_df = pd.read_csv(INPUT_DATA_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    return predictions_df, input_data_df, train_df


def extract_ground_truth(train_df):
    """train.csv에서 ground truth answer 추출"""
    ground_truth = {}
    for _, row in train_df.iterrows():
        problems = literal_eval(row["problems"])
        answer = problems.get("answer", None)
        ground_truth[row["id"]] = answer
    return ground_truth


def calculate_macro_f1(y_true, y_pred):
    """Macro F1-score 계산"""
    return f1_score(y_true, y_pred, average="macro", labels=[1, 2, 3, 4, 5])


def calculate_per_class_f1(y_true, y_pred):
    """클래스별 F1-score 계산"""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2, 3, 4, 5], zero_division=0
    )
    return {"class": [1, 2, 3, 4, 5], "precision": precision, "recall": recall, "f1": f1, "support": support}


def analyze_errors(predictions_df, input_data_df, ground_truth):
    """에러 케이스 분석"""
    errors = []
    correct = []

    for _, row in predictions_df.iterrows():
        id_ = row["id"]
        pred = row["answer"]
        true = ground_truth.get(id_, None)

        if true is None:
            continue

        if pred != true:
            # input_data에서 해당 문제 정보 가져오기
            input_row = input_data_df[input_data_df["id"] == id_]
            if len(input_row) > 0:
                input_row = input_row.iloc[0]
                errors.append(
                    {
                        "id": id_,
                        "predicted": pred,
                        "ground_truth": true,
                        "paragraph": input_row.get("paragraph", "")[:200] + "..."
                        if len(str(input_row.get("paragraph", ""))) > 200
                        else input_row.get("paragraph", ""),
                        "question": input_row.get("question", ""),
                        "choices": input_row.get("choices", ""),
                    }
                )
        else:
            correct.append({"id": id_, "predicted": pred, "ground_truth": true})

    return errors, correct


def print_confusion_matrix(y_true, y_pred):
    """Confusion Matrix 출력"""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    print("\n=== Confusion Matrix ===")
    print("(행: 실제, 열: 예측)")
    print("       Pred_1  Pred_2  Pred_3  Pred_4  Pred_5")
    for i, row in enumerate(cm):
        print(f"True_{i + 1}  {row[0]:5d}   {row[1]:5d}   {row[2]:5d}   {row[3]:5d}   {row[4]:5d}")
    return cm


def analyze_error_patterns(errors):
    """에러 패턴 분석"""
    if not errors:
        return {}

    error_df = pd.DataFrame(errors)

    # 예측값별 에러 분포
    pred_dist = error_df["predicted"].value_counts().sort_index()

    # 실제값별 에러 분포
    true_dist = error_df["ground_truth"].value_counts().sort_index()

    # 예측 -> 실제 전환 패턴
    transition = error_df.groupby(["predicted", "ground_truth"]).size().reset_index(name="count")
    transition = transition.sort_values("count", ascending=False)

    return {"pred_distribution": pred_dist, "true_distribution": true_dist, "transition_patterns": transition}


def main():
    print("=" * 60)
    print("Train 데이터 Error Case Analysis & Scoring")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    predictions_df, input_data_df, train_df = load_data()
    print(f"    - 예측 결과 수: {len(predictions_df)}")
    print(f"    - 입력 데이터 수: {len(input_data_df)}")
    print(f"    - 학습 데이터 수: {len(train_df)}")

    # 2. Ground Truth 추출
    print("\n[2] Ground Truth 추출 중...")
    ground_truth = extract_ground_truth(train_df)
    print(f"    - Ground Truth 추출 완료: {len(ground_truth)}개")

    # 3. 예측값과 실제값 매칭
    print("\n[3] 예측값과 실제값 매칭 중...")
    y_true = []
    y_pred = []
    matched_ids = []

    for _, row in predictions_df.iterrows():
        id_ = row["id"]
        if id_ in ground_truth and ground_truth[id_] is not None:
            y_true.append(ground_truth[id_])
            y_pred.append(row["answer"])
            matched_ids.append(id_)

    print(f"    - 매칭된 샘플 수: {len(y_true)}")

    # 4. 클래스 분포 확인
    print("\n[4] 클래스 분포")
    print("    Ground Truth 분포:")
    gt_dist = pd.Series(y_true).value_counts().sort_index()
    for cls, cnt in gt_dist.items():
        print(f"        클래스 {cls}: {cnt}개 ({cnt / len(y_true) * 100:.1f}%)")

    print("\n    예측값 분포:")
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    for cls, cnt in pred_dist.items():
        print(f"        클래스 {cls}: {cnt}개 ({cnt / len(y_pred) * 100:.1f}%)")

    # 5. Scoring
    print("\n" + "=" * 60)
    print("[5] SCORING RESULTS")
    print("=" * 60)

    # Accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    print(f"\n    Accuracy: {accuracy:.4f} ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})")

    # Macro F1-score
    macro_f1 = calculate_macro_f1(y_true, y_pred)
    print(f"    Macro F1-score: {macro_f1:.4f}")

    # 클래스별 F1-score
    print("\n    클래스별 성능:")
    per_class = calculate_per_class_f1(y_true, y_pred)
    print("    " + "-" * 50)
    print(f"    {'클래스':^6} {'Precision':^10} {'Recall':^10} {'F1':^10} {'Support':^8}")
    print("    " + "-" * 50)
    for i in range(5):
        print(
            f"    {per_class['class'][i]:^6} {per_class['precision'][i]:^10.4f} {per_class['recall'][i]:^10.4f} {per_class['f1'][i]:^10.4f} {int(per_class['support'][i]):^8}"
        )
    print("    " + "-" * 50)

    # Confusion Matrix
    cm = print_confusion_matrix(y_true, y_pred)

    # 6. Error Case Analysis
    print("\n" + "=" * 60)
    print("[6] ERROR CASE ANALYSIS")
    print("=" * 60)

    errors, correct = analyze_errors(predictions_df, input_data_df, ground_truth)
    print(f"\n    총 에러 수: {len(errors)}")
    print(f"    정답 수: {len(correct)}")

    # 에러 패턴 분석
    error_patterns = analyze_error_patterns(errors)

    if error_patterns:
        print("\n    [에러 패턴 분석]")
        print("\n    예측값별 에러 분포 (모델이 잘못 예측한 답):")
        for cls, cnt in error_patterns["pred_distribution"].items():
            print(f"        클래스 {cls}: {cnt}개")

        print("\n    실제값별 에러 분포 (모델이 틀린 문제의 정답):")
        for cls, cnt in error_patterns["true_distribution"].items():
            print(f"        클래스 {cls}: {cnt}개")

        print("\n    주요 오분류 패턴 (Top 10):")
        print("    " + "-" * 40)
        print(f"    {'예측':^6} {'실제':^6} {'빈도':^8}")
        print("    " + "-" * 40)
        for _, row in error_patterns["transition_patterns"].head(10).iterrows():
            print(f"    {int(row['predicted']):^6} {int(row['ground_truth']):^6} {int(row['count']):^8}")

    # 7. 상세 에러 케이스 샘플
    print("\n" + "=" * 60)
    print("[7] ERROR CASE SAMPLES (처음 10개)")
    print("=" * 60)

    for i, error in enumerate(errors[:10]):
        print(f"\n--- Error Case #{i + 1} ---")
        print(f"ID: {error['id']}")
        print(f"예측: {error['predicted']} / 정답: {error['ground_truth']}")
        print(f"질문: {error['question']}")
        print(f"선택지: {error['choices']}")
        print(f"지문: {error['paragraph']}")

    # 8. 요약
    print("\n" + "=" * 60)
    print("[8] SUMMARY")
    print("=" * 60)
    print(f"""
    - 총 평가 샘플: {len(y_true)}개
    - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
    - Macro F1-score: {macro_f1:.4f}
    - 에러 케이스: {len(errors)}개
    - 정답 케이스: {len(correct)}개

    [클래스별 F1-score]
    - 클래스 1: {per_class["f1"][0]:.4f}
    - 클래스 2: {per_class["f1"][1]:.4f}
    - 클래스 3: {per_class["f1"][2]:.4f}
    - 클래스 4: {per_class["f1"][3]:.4f}
    - 클래스 5: {per_class["f1"][4]:.4f}
    """)

    # 9. 에러 케이스를 CSV로 저장
    error_df = pd.DataFrame(errors)
    # error_output_path = "/data/ephemeral/home/kdh/outputs/qwen3_2507_thinking_train_0102_221623/error_cases.csv"
    time_stamp = date.now().strftime("%m%d_%H%M%S")
    error_output_path = f"/data/ephemeral/home/kdh/error_cases_{time_stamp}.csv"
    error_df.to_csv(error_output_path, index=False)
    print(f"\n에러 케이스가 저장되었습니다: {error_output_path}")


if __name__ == "__main__":
    main()
