"""
앙상블 스크립트 for Macro F1-score 최적화

가중치 투표 방식 설명:
- 각 선택지(1~5)에 대해 "투표 점수"를 계산
- 모델이 어떤 선택지를 예측하면, 그 선택지에 해당 모델의 가중치만큼 점수 추가
- 가장 높은 점수를 받은 선택지를 최종 답으로 선택

예시:
문제 ID: generation-for-nlp-0
모델 예측:
  - qwen3_thinking: 4 (가중치 0.4)
  - qwen3_instruct: 4 (가중치 0.2)
  - qwen2.5_haerae: 4 (가중치 0.15)
  - qwen2.5_shuffle: 5 (가중치 0.15)
  - exaone: 3 (가중치 0.1)

선택지별 점수 계산:
  - 선택지 3: 0.1 (exaone만 선택)
  - 선택지 4: 0.4 + 0.2 + 0.15 = 0.75 (3개 모델 선택)
  - 선택지 5: 0.15 (shuffle만 선택)

최종 답: 4 (가장 높은 점수 0.75)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from datetime import datetime
import argparse
import itertools


class EnsembleModel:
    def __init__(self, csv_dir: str):
        """
        Args:
            csv_dir: CSV 파일들이 있는 디렉토리
        """
        self.csv_dir = Path(csv_dir)
        self.dir_name = self.csv_dir.name  # 디렉토리명 저장
        self.models = {}
        self.model_names = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_predictions(self, model_files: Dict[str, str]):
        """
        모델 예측 결과 로드

        Args:
            model_files: {모델명: 파일명} 딕셔너리
        """
        print("="*80)
        print("모델 로딩")
        print("="*80)

        for model_name, filename in model_files.items():
            filepath = self.csv_dir / filename
            df = pd.read_csv(filepath)
            self.models[model_name] = df
            self.model_names.append(model_name)
            print(f"✓ {model_name:20s}: {len(df):4d} predictions")

        # ID 일치 확인
        ids = [set(df['id']) for df in self.models.values()]
        if len(set(map(len, ids))) != 1:
            print("\n⚠ Warning: 모델들의 예측 개수가 다릅니다!")
            for model_name in self.model_names:
                print(f"  {model_name}: {len(self.models[model_name])} samples")

        common_ids = set.intersection(*ids)
        print(f"\n✓ 공통 ID 개수: {len(common_ids)}")

    def analyze_predictions(self, save_analysis: bool = True):
        """예측 결과 심층 분석"""
        print("\n" + "="*80)
        print("예측 심층 분석")
        print("="*80)

        analysis_lines = []
        analysis_lines.append("="*80)
        analysis_lines.append("예측 심층 분석")
        analysis_lines.append("="*80)

        # 1. 각 모델의 클래스 분포
        print("\n[1. 클래스 분포]")
        analysis_lines.append("\n[1. 클래스 분포]")
        for model_name in self.model_names:
            dist = self.models[model_name]['answer'].value_counts().sort_index()
            line = f"{model_name:20s}: {dict(dist)}"
            print(line)
            analysis_lines.append(line)

        # 2. 모델 간 일치도 행렬
        print("\n[2. 모델 간 일치도 행렬]")
        analysis_lines.append("\n[2. 모델 간 일치도 행렬]")

        agreement_matrix = pd.DataFrame(
            np.zeros((len(self.model_names), len(self.model_names))),
            index=self.model_names,
            columns=self.model_names
        )

        for i, model1 in enumerate(self.model_names):
            for j, model2 in enumerate(self.model_names):
                if i == j:
                    agreement_matrix.iloc[i, j] = 100.0
                else:
                    df1 = self.models[model1]
                    df2 = self.models[model2]
                    merged = df1.merge(df2, on='id', suffixes=('_1', '_2'))
                    agreement = (merged['answer_1'] == merged['answer_2']).sum()
                    agreement_pct = agreement / len(merged) * 100
                    agreement_matrix.iloc[i, j] = agreement_pct

        print(agreement_matrix.round(1).to_string())
        analysis_lines.append(agreement_matrix.round(1).to_string())

        # 3. 전체 모델 합의 분석
        print("\n[3. 모델 합의 분석]")
        analysis_lines.append("\n[3. 모델 합의 분석]")

        first_model = self.models[self.model_names[0]]
        df_merged = first_model[['id', 'answer']].copy()
        df_merged.columns = ['id', self.model_names[0]]

        for model_name in self.model_names[1:]:
            df_merged = df_merged.merge(
                self.models[model_name][['id', 'answer']],
                on='id',
                how='inner'
            )
            df_merged.columns = list(df_merged.columns[:-1]) + [model_name]

        # 각 문제별 합의 모델 수 계산
        answer_cols = self.model_names
        df_merged['agreement_count'] = df_merged[answer_cols].apply(
            lambda row: row.value_counts().max(), axis=1
        )

        agreement_dist = df_merged['agreement_count'].value_counts().sort_index()
        print("합의 모델 수 분포:")
        analysis_lines.append("합의 모델 수 분포:")
        for count, freq in agreement_dist.items():
            pct = freq / len(df_merged) * 100
            line = f"  {count}개 모델 합의: {freq:4d}개 ({pct:5.1f}%)"
            print(line)
            analysis_lines.append(line)

        # 4. 불일치 케이스 상세 분석
        print("\n[4. 불일치 케이스 분석]")
        analysis_lines.append("\n[4. 불일치 케이스 분석]")

        # 완전 불일치 (모든 모델이 다른 답)
        all_different = df_merged[answer_cols].nunique(axis=1) == len(self.model_names)
        print(f"완전 불일치 (5개 모델 모두 다른 답): {all_different.sum()}개")
        analysis_lines.append(f"완전 불일치 (5개 모델 모두 다른 답): {all_different.sum()}개")

        if all_different.sum() > 0:
            print("  샘플 예시 (최대 5개):")
            analysis_lines.append("  샘플 예시 (최대 5개):")
            for idx in df_merged[all_different].head(5).index:
                row = df_merged.loc[idx]
                line = f"    {row['id']}: " + ", ".join([f"{m}={row[m]}" for m in self.model_names])
                print(line)
                analysis_lines.append(line)

        # 5. 모델별 고유 예측 분석 (다른 모델과 다르게 예측한 케이스)
        print("\n[5. 모델별 고유 예측 분석]")
        analysis_lines.append("\n[5. 모델별 고유 예측 분석]")

        for model_name in self.model_names:
            other_models = [m for m in self.model_names if m != model_name]

            # 해당 모델이 다른 모든 모델과 다르게 예측한 케이스
            unique_pred = df_merged.apply(
                lambda row: all(row[model_name] != row[other] for other in other_models),
                axis=1
            )

            unique_count = unique_pred.sum()
            pct = unique_count / len(df_merged) * 100
            line = f"{model_name:20s}: {unique_count:4d}개 ({pct:5.1f}%) - 혼자만 다른 예측"
            print(line)
            analysis_lines.append(line)

        # 6. Thinking 모델 vs 다른 모델들 비교
        if 'qwen3_thinking' in self.model_names:
            print("\n[6. Thinking 모델 특별 분석]")
            analysis_lines.append("\n[6. Thinking 모델 특별 분석]")

            thinking_col = 'qwen3_thinking'
            other_models = [m for m in self.model_names if m != thinking_col]

            # Thinking만 다르게 예측한 케이스
            thinking_alone = df_merged.apply(
                lambda row: all(row[thinking_col] != row[other] for other in other_models),
                axis=1
            )

            # 다른 모델들은 모두 같은데 Thinking만 다른 케이스
            others_agree_thinking_diff = df_merged.apply(
                lambda row: (len(set(row[other_models])) == 1) and (row[thinking_col] != row[other_models[0]]),
                axis=1
            )

            line1 = f"Thinking이 다른 모든 모델과 다른 예측: {thinking_alone.sum()}개"
            line2 = f"다른 4개 모델은 합의했으나 Thinking만 다른 예측: {others_agree_thinking_diff.sum()}개"
            print(line1)
            print(line2)
            analysis_lines.append(line1)
            analysis_lines.append(line2)

            if others_agree_thinking_diff.sum() > 0:
                print("  샘플 예시 (최대 10개):")
                analysis_lines.append("  샘플 예시 (최대 10개):")
                for idx in df_merged[others_agree_thinking_diff].head(10).index:
                    row = df_merged.loc[idx]
                    others_answer = row[other_models[0]]
                    line = f"    {row['id']}: Thinking={row[thinking_col]} vs Others={others_answer}"
                    print(line)
                    analysis_lines.append(line)

        # 7. 클래스별 모델 예측 분포
        print("\n[7. 클래스별 예측 패턴]")
        analysis_lines.append("\n[7. 클래스별 예측 패턴]")

        for cls in range(1, 6):
            print(f"\n클래스 {cls}를 예측한 문제들의 모델별 분포:")
            analysis_lines.append(f"\n클래스 {cls}를 예측한 문제들의 모델별 분포:")

            for model_name in self.model_names:
                count = (self.models[model_name]['answer'] == cls).sum()
                pct = count / len(self.models[model_name]) * 100
                line = f"  {model_name:20s}: {count:4d}개 ({pct:5.1f}%)"
                print(line)
                analysis_lines.append(line)

        # 분석 결과 저장
        if save_analysis:
            analysis_file = self.csv_dir / f"analysis_{self.dir_name}_{self.timestamp}.txt"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(analysis_lines))
            print(f"\n✓ 분석 결과 저장: {analysis_file}")

    def weighted_voting(self, weights: Dict[str, float]) -> pd.DataFrame:
        """
        가중치 투표 앙상블

        Args:
            weights: {모델명: 가중치} 딕셔너리

        Returns:
            앙상블 결과 DataFrame (id, answer)
        """
        print(f"\n[가중치 투표]")
        for model, weight in weights.items():
            print(f"  {model:20s}: {weight:.2f}")

        # 첫 번째 모델의 ID를 기준으로
        result_df = self.models[self.model_names[0]][['id']].copy()
        predictions = []

        for idx, row in result_df.iterrows():
            id_val = row['id']

            # 각 선택지(1~5)에 대한 점수 계산
            scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

            for model_name in self.model_names:
                model_df = self.models[model_name]
                pred = model_df[model_df['id'] == id_val]['answer'].values[0]
                weight = weights.get(model_name, 0.0)
                scores[pred] += weight

            # 가장 높은 점수의 선택지 선택
            final_answer = max(scores, key=scores.get)
            predictions.append(final_answer)

        result_df['answer'] = predictions
        return result_df

    def hard_voting(self, tie_breaker_model: str = None) -> pd.DataFrame:
        """
        하드 투표 (다수결)

        Args:
            tie_breaker_model: 동점일 때 우선할 모델 (None이면 첫 번째 예측 사용)

        Returns:
            앙상블 결과 DataFrame
        """
        print(f"\n[하드 투표] tie_breaker={tie_breaker_model}")

        result_df = self.models[self.model_names[0]][['id']].copy()
        predictions = []
        tie_count = 0

        for idx, row in result_df.iterrows():
            id_val = row['id']

            # 모든 모델의 예측 수집
            votes = []
            for model_name in self.model_names:
                model_df = self.models[model_name]
                pred = model_df[model_df['id'] == id_val]['answer'].values[0]
                votes.append(pred)

            # 다수결
            vote_counts = Counter(votes)
            max_count = max(vote_counts.values())
            candidates = [ans for ans, count in vote_counts.items() if count == max_count]

            if len(candidates) == 1:
                final_answer = candidates[0]
            else:
                # 동점일 경우
                tie_count += 1
                if tie_breaker_model:
                    tie_breaker_pred = self.models[tie_breaker_model][
                        self.models[tie_breaker_model]['id'] == id_val
                    ]['answer'].values[0]
                    final_answer = tie_breaker_pred if tie_breaker_pred in candidates else candidates[0]
                else:
                    final_answer = candidates[0]

            predictions.append(final_answer)

        print(f"동점 발생: {tie_count}/{len(result_df)} ({tie_count/len(result_df)*100:.1f}%)")
        result_df['answer'] = predictions
        return result_df

    def thinking_priority_voting(self, thinking_model: str, threshold: int = 2) -> pd.DataFrame:
        """
        Thinking 모델 우선 투표
        - threshold 이상의 모델이 동의하면 그 답 선택
        - 그렇지 않으면 thinking 모델 선택

        Args:
            thinking_model: 우선할 thinking 모델명
            threshold: 최소 동의 모델 수
        """
        print(f"\n[Thinking 우선 투표] thinking_model={thinking_model}, threshold={threshold}")

        result_df = self.models[self.model_names[0]][['id']].copy()
        predictions = []
        thinking_used = 0

        for idx, row in result_df.iterrows():
            id_val = row['id']

            # 모든 모델의 예측 수집
            votes = []
            for model_name in self.model_names:
                model_df = self.models[model_name]
                pred = model_df[model_df['id'] == id_val]['answer'].values[0]
                votes.append(pred)

            # 다수결 확인
            vote_counts = Counter(votes)
            max_count = max(vote_counts.values())

            if max_count >= threshold:
                # threshold 이상 동의하는 답이 있으면 그것 선택
                final_answer = max(vote_counts, key=vote_counts.get)
            else:
                # 명확한 다수가 없으면 thinking 모델 선택
                final_answer = self.models[thinking_model][
                    self.models[thinking_model]['id'] == id_val
                ]['answer'].values[0]
                thinking_used += 1

            predictions.append(final_answer)

        print(f"Thinking 모델 사용: {thinking_used}/{len(result_df)} ({thinking_used/len(result_df)*100:.1f}%)")
        result_df['answer'] = predictions
        return result_df

    def save_submission(self, df: pd.DataFrame, base_filename: str, method_info: str = ""):
        """
        제출 파일 저장 (디렉토리명 + 타임스탬프 포함)

        Args:
            df: 저장할 DataFrame
            base_filename: 기본 파일명 (예: 'ensemble_weighted')
            method_info: 추가 메서드 정보 (예: 가중치 정보)
        """
        # 파일명 생성: base_name_dirName_timestamp.csv
        output_filename = f"{base_filename}_{self.dir_name}_{self.timestamp}.csv"
        output_path = self.csv_dir / output_filename

        df.to_csv(output_path, index=False)

        class_dist = dict(df['answer'].value_counts().sort_index())
        print(f"\n✓ 저장: {output_filename}")
        print(f"  클래스 분포: {class_dist}")
        if method_info:
            print(f"  {method_info}")


def main():
    parser = argparse.ArgumentParser(
        description='앙상블 스크립트 - Macro F1-score 최적화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 모든 방법 실행
  python ensemble.py

  # 특정 방법만 실행
  python ensemble.py --method weighted

  # Thinking threshold 변경
  python ensemble.py --method thinking_priority --threshold 4

  # 여러 가중치 조합 실험
  python ensemble.py --method weighted --experiment

  # 분석만 수행
  python ensemble.py --analyze-only
        """
    )
    parser.add_argument('--dir', type=str, default='/data/ephemeral/home/kdh/ensemble/sotas_0105_2229',
                        help='CSV 파일 디렉토리')
    parser.add_argument('--method', type=str, default='all',
                        choices=['weighted', 'hard', 'thinking_priority', 'all'],
                        help='앙상블 방법')
    parser.add_argument('--threshold', type=int, default=3,
                        help='Thinking priority voting의 threshold (기본값: 3)')
    parser.add_argument('--experiment', action='store_true',
                        help='여러 가중치 조합 실험 (weighted 방법 전용)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='분석만 수행하고 앙상블은 수행하지 않음')
    args = parser.parse_args()

    # 모델 파일 정의
    model_files = {
        'qwen3_thinking': 'qwen3_thinking.csv',
        'qwen3_instruct': 'qwen3_instruct.csv',
        'qwen2.5_haerae': 'qwen2.5_haerae.csv',
        'qwen2.5_shuffle': 'qwen2.5_shuffle.csv',
        'exaone': 'exaone.csv'
    }

    # 앙상블 모델 초기화
    ensemble = EnsembleModel(args.dir)
    ensemble.load_predictions(model_files)

    # 예측 분석
    ensemble.analyze_predictions(save_analysis=True)

    if args.analyze_only:
        print("\n✓ 분석 완료 (--analyze-only 옵션)")
        return

    print("\n" + "="*80)
    print("앙상블 실행")
    print("="*80)

    if args.method == 'weighted' or args.method == 'all':
        if args.experiment:
            # 여러 가중치 조합 실험
            print("\n🔬 가중치 조합 실험 모드")

            weight_configs = [
                # Thinking 모델 가중치를 다양하게 실험
                {'qwen3_thinking': 0.5, 'qwen3_instruct': 0.2, 'qwen2.5_haerae': 0.1, 'qwen2.5_shuffle': 0.1, 'exaone': 0.1},
                {'qwen3_thinking': 0.4, 'qwen3_instruct': 0.2, 'qwen2.5_haerae': 0.15, 'qwen2.5_shuffle': 0.15, 'exaone': 0.1},
                {'qwen3_thinking': 0.35, 'qwen3_instruct': 0.25, 'qwen2.5_haerae': 0.15, 'qwen2.5_shuffle': 0.15, 'exaone': 0.1},
                {'qwen3_thinking': 0.3, 'qwen3_instruct': 0.25, 'qwen2.5_haerae': 0.15, 'qwen2.5_shuffle': 0.15, 'exaone': 0.15},
            ]

            for i, weights in enumerate(weight_configs, 1):
                result = ensemble.weighted_voting(weights)
                weight_str = "_".join([f"{k.split('_')[-1]}{int(v*100)}" for k, v in weights.items()])
                ensemble.save_submission(result, f'ensemble_weighted_exp{i}_{weight_str}')
        else:
            # 기본 가중치 설정 (thinking 모델 우선)
            weights = {
                'qwen3_thinking': 0.4,
                'qwen3_instruct': 0.2,
                'qwen2.5_haerae': 0.15,
                'qwen2.5_shuffle': 0.15,
                'exaone': 0.1
            }
            result = ensemble.weighted_voting(weights)
            ensemble.save_submission(result, 'ensemble_weighted')

    if args.method == 'hard' or args.method == 'all':
        # 하드 투표 (thinking 모델을 tie-breaker로)
        result = ensemble.hard_voting(tie_breaker_model='qwen3_thinking')
        ensemble.save_submission(result, 'ensemble_hard_voting')

    if args.method == 'thinking_priority' or args.method == 'all':
        # Thinking 우선 투표 (커맨드라인에서 threshold 조정 가능)
        result = ensemble.thinking_priority_voting('qwen3_thinking', threshold=args.threshold)
        ensemble.save_submission(result, f'ensemble_thinking_priority_th{args.threshold}')

    print("\n" + "="*80)
    print("✅ 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
