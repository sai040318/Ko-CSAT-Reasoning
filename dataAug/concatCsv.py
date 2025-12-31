import pandas as pd
import os

# 파일 경로 설정
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
train_path = os.path.join(data_dir, 'train.csv')
trans_path = os.path.join(data_dir, 'self_instruct_output.csv')
output_path = os.path.join(data_dir, 'train_self.csv')

# CSV 파일 읽기
print(f"Loading {train_path}...")
train_df = pd.read_csv(train_path)
print(f"train.csv: {len(train_df)} rows")

print(f"Loading {trans_path}...")
trans_df = pd.read_csv(trans_path)
print(f"trans2.csv: {len(trans_df)} rows")

# 데이터 합치기
concat_df = pd.concat([train_df, trans_df], ignore_index=True)
print(f"\nConcatenated: {len(concat_df)} rows")

# 결과 저장
concat_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
print(f"\nColumns: {list(concat_df.columns)}")
print(f"Total rows: {len(concat_df)}")
