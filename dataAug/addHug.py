import pandas as pd
import os
from pathlib import Path
from datasets import load_dataset

# 스크립트 파일 위치를 기준으로 프로젝트 루트 경로 설정
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
data_dir = project_root / "data"

# 1. 허깅페이스 데이터셋 불러오기
print("허깅페이스 데이터 로딩 중...")

# 사용 가능한 config 목록
# available_configs = ['General Knowledge', 'History', 'Loan Words', 'Reading Comprehension', 'Rare Words', 'Standard Nomenclature']
available_configs = ['History', 'Reading Comprehension']

# 모든 config를 로드하여 병합
print(f"사용 가능한 config: {available_configs}")
print("모든 config를 로드 중...")

# 각 데이터셋을 DataFrame으로 변환하여 리스트에 저장
df_list = []
for config_name in available_configs:
    print(f"  - {config_name} 로딩 중...")
    try:
        # 이 데이터셋은 'test' split만 제공됨
        config_data = load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.0", config_name, split="test")
        # 각 데이터셋을 개별적으로 DataFrame으로 변환 (feature 타입 불일치 문제 해결)
        df_config = config_data.to_pandas()
        df_list.append(df_config)
        print(f"    ✅ {len(df_config)}개 샘플 로드 완료")
    except Exception as e:
        print(f"    ⚠️ {config_name} 로드 실패: {e}")

# 모든 DataFrame을 하나로 병합
if df_list:
    print("\nPandas DataFrame으로 병합 중...")
    df_hf = pd.concat(df_list, ignore_index=True)
    print(f"총 {len(df_hf)}개 샘플 로드 완료")
else:
    raise ValueError("로드된 데이터셋이 없습니다.")

# 3. 내 로컬 데이터 불러오기 (컬럼 구조 확인용)
print("내 데이터 로딩 중...")
train_csv_path = data_dir / "train.csv"
print(f"데이터 경로: {train_csv_path}")
df_local = pd.read_csv(train_csv_path)

# 4. train.csv의 컬럼 구조 확인
print("\n=== train.csv 컬럼 구조 ===")
print("컬럼명:", df_local.columns.tolist())
print(f"샘플 수: {len(df_local)}")

# 5. Hugging Face 데이터셋의 컬럼 구조 확인
print("\n=== Hugging Face 데이터셋 컬럼 구조 ===")
print("컬럼명:", df_hf.columns.tolist())
print(f"샘플 수: {len(df_hf)}")
print("\n첫 번째 샘플:")
print(df_hf.head(1))
print("\n데이터 타입:")
print(df_hf.dtypes)
print("\n각 컬럼별 샘플 값 (처음 3개):")
for col in df_hf.columns:
    print(f"  {col}: {df_hf[col].head(3).tolist()}")

# 6. train.csv 컬럼 구조에 맞게 데이터셋 컬럼 수정
print("\n=== 컬럼 매핑 및 변환 ===")

# train.csv의 컬럼: id, paragraph, problems, question_plus
target_columns = df_local.columns.tolist()

# Hugging Face 데이터셋의 컬럼을 train.csv 구조에 맞게 매핑
# (실제 데이터셋 구조에 따라 수정 필요)
df_hf_mapped = pd.DataFrame()

# id 컬럼 생성 (없으면 인덱스 기반으로 생성)
if 'id' in df_hf.columns:
    df_hf_mapped['id'] = df_hf['id']
else:
    df_hf_mapped['id'] = [f"hf-{i}" for i in range(len(df_hf))]

# paragraph 컬럼 매핑 (context, passage, text 등 우선순위로 찾기)
# 주의: question이 paragraph에 들어가지 않도록 해야 함
paragraph_candidates = ['paragraph', 'context', 'passage', 'text', 'reference', '참고']
paragraph_col = None

for candidate in paragraph_candidates:
    if candidate in df_hf.columns:
        paragraph_col = candidate
        break

if paragraph_col:
    df_hf_mapped['paragraph'] = df_hf[paragraph_col]
else:
    # paragraph에 해당하는 컬럼이 없으면 빈 문자열로 설정
    # (일부 데이터셋은 paragraph가 없을 수 있음)
    print("⚠️ paragraph 컬럼을 찾을 수 없어 빈 문자열로 설정합니다.")
    df_hf_mapped['paragraph'] = ''

# problems 컬럼 매핑 (딕셔너리 문자열 형태로 변환)
def create_problems_str(row):
    """row에서 question, choices, answer 정보를 추출하여 problems 딕셔너리 생성"""
    problems_dict = {}
    
    # question 추출
    if 'question' in df_hf.columns:
        problems_dict['question'] = str(row.get('question', ''))
    else:
        problems_dict['question'] = ''
    
    # choices 추출 (여러 방법 시도)
    choices = []
    if 'choices' in df_hf.columns:
        # choices가 리스트/튜플인 경우
        choices_val = row.get('choices', [])
        if isinstance(choices_val, (list, tuple)):
            choices = [str(c) for c in choices_val if pd.notna(c)]
        elif pd.notna(choices_val):
            choices = [str(choices_val)]
    elif all(col in df_hf.columns for col in ['a', 'b', 'c', 'd']):
        # a, b, c, d 컬럼이 있는 경우
        choices = []
        for choice_col in ['a', 'b', 'c', 'd']:
            val = row.get(choice_col, '')
            if pd.notna(val) and str(val).strip():
                choices.append(str(val))
        # e 컬럼이 있으면 추가
        if 'e' in df_hf.columns:
            e_val = row.get('e', '')
            if pd.notna(e_val) and str(e_val).strip() and str(e_val) != 'nan':
                choices.append(str(e_val))
    elif 'option1' in df_hf.columns:
        # option1, option2 등이 있는 경우
        option_cols = [col for col in df_hf.columns if col.startswith('option')]
        option_cols.sort()
        choices = [str(row.get(col, '')) for col in option_cols if pd.notna(row.get(col, '')) and str(row.get(col, '')).strip()]
    
    problems_dict['choices'] = choices
    
    # answer 추출
    if 'answer' in df_hf.columns:
        answer_val = row.get('answer', 0)
        # answer가 문자열인 경우 숫자로 변환 시도
        if isinstance(answer_val, str):
            # 'a', 'b', 'c', 'd', 'e'를 인덱스로 변환
            answer_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
            problems_dict['answer'] = answer_map.get(answer_val.lower(), 0)
        else:
            problems_dict['answer'] = int(answer_val) if pd.notna(answer_val) else 0
    elif 'e' in df_hf.columns:
        # e 컬럼이 float64인 경우 (인덱스로 사용)
        e_val = row.get('e', 0)
        problems_dict['answer'] = int(e_val) if pd.notna(e_val) else 0
    else:
        problems_dict['answer'] = 0
    
    # train.csv와 동일한 형식으로 변환 (딕셔너리 문자열)
    return str(problems_dict)

if 'problems' in df_hf.columns:
    # 이미 problems 컬럼이 있으면 그대로 사용
    df_hf_mapped['problems'] = df_hf['problems'].astype(str)
else:
    # row별로 problems 생성
    df_hf_mapped['problems'] = df_hf.apply(create_problems_str, axis=1)

# question_plus 컬럼 매핑
if 'question_plus' in df_hf.columns:
    df_hf_mapped['question_plus'] = df_hf['question_plus'].astype(str)
else:
    # question_plus가 없으면 problems와 동일하게 설정
    df_hf_mapped['question_plus'] = None

# 7. 최종 결과 확인
print("\n=== 변환된 데이터셋 구조 ===")
print("컬럼명:", df_hf_mapped.columns.tolist())
print(f"샘플 수: {len(df_hf_mapped)}")
print("\n첫 번째 샘플:")
print(df_hf_mapped.head(1))

# 8. train.csv와 동일한 컬럼 순서로 정렬
df_hf_mapped = df_hf_mapped[target_columns]

print("\n=== 최종 확인 ===")
print("컬럼 순서:", df_hf_mapped.columns.tolist())
print("train.csv와 동일한 구조:", df_hf_mapped.columns.tolist() == target_columns)

# 9. 두 데이터프레임 병합
print("\n=== 데이터 병합 ===")
print(f"로컬 데이터 샘플 수: {len(df_local)}")
print(f"Hugging Face 데이터 샘플 수: {len(df_hf_mapped)}")

# 두 데이터프레임을 세로로 병합 (concat)
df_merged = pd.concat([df_local, df_hf_mapped], ignore_index=True)

print(f"병합된 데이터 샘플 수: {len(df_merged)}")
print(f"병합 후 컬럼: {df_merged.columns.tolist()}")

# 10. 병합된 데이터를 새로운 CSV 파일로 저장
output_path = data_dir / "train_his_read.csv"
print(f"\n=== CSV 파일 저장 ===")
print(f"저장 경로: {output_path}")
df_merged.to_csv(output_path, index=False, encoding='utf-8')
print(f"✅ 병합된 데이터가 '{output_path}'에 저장되었습니다!")
print(f"   - 총 {len(df_merged)}개의 샘플")
print(f"   - 컬럼: {df_merged.columns.tolist()}")

