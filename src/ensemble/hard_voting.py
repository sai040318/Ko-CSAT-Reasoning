import pandas as pd
from collections import Counter
import os

# inference CSV 파일 경로
csv_files = [
    'qwen3_thinking.csv',
    'qwen3_instruct.csv',
    'qwen2.5_32b_output.csv',
    'qwen2.5_haerae.csv',
    'qwen2.5_base.csv',
]

script_dir = os.path.dirname(os.path.abspath(__file__))

# 모든 CSV 파일 읽기
dataframes = {}
for csv_file in csv_files:
    file_path = os.path.join(script_dir, csv_file)
    df = pd.read_csv(file_path)
    dataframes[csv_file] = df
    print(f"Loaded {csv_file}: {len(df)} rows")

# qwen3_thinking.csv를 tie-breaker로 사용
# 가장 성능이 우수
tie_breaker_df = dataframes['qwen3_thinking.csv'].set_index('id')

merged_df = dataframes['qwen3_thinking.csv'][['id']].copy()

for csv_file in csv_files:
    df = dataframes[csv_file].set_index('id')
    merged_df[csv_file] = merged_df['id'].map(df['answer'])

# Hard voting 수행
def hard_vote(row):
    votes = []
    for csv_file in csv_files:
        vote = row[csv_file]
        if pd.notna(vote):  
            votes.append(int(vote))
    # 다수결 계산
    vote_counts = Counter(votes)
    max_count = max(vote_counts.values())
    
    # 최빈값이 여러 개인 경우 (동점)
    most_common = [vote for vote, count in vote_counts.items() if count == max_count]
    
    if len(most_common) > 1:
        # 타이브레이커
        tie_breaker_answer = tie_breaker_df.loc[row['id'], 'answer']
        if pd.notna(tie_breaker_answer):
            return int(tie_breaker_answer)

# Hard voting 적용
merged_df['answer'] = merged_df.apply(hard_vote, axis=1)

# 결과 저장
output_df = merged_df[['id', 'answer']].copy()
output_path = os.path.join(script_dir, 'hard_voting_ensemble.csv')
output_df.to_csv(output_path, index=False)

print(f"\nHard voting 완료!")

# 통계 출력
print("\n=== 통계 ===")
for csv_file in csv_files:
    agreement = (merged_df['answer'] == merged_df[csv_file]).sum()
    agreement_pct = (agreement / len(merged_df)) * 100
    print(f"{csv_file}: {agreement}개 일치 ({agreement_pct:.2f}%)")

