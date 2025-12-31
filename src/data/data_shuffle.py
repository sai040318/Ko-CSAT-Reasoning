import pandas as pd
import random
from ast import literal_eval

# 시드 값 고정
SEED = 42
random.seed(SEED)

# 데이터 load
df = pd.read_csv('data/train.csv')

# Flatten the JSON df
records = []
for _, row in df.iterrows():
    problems = literal_eval(row['problems']) 
    
    choices = problems['choices']
    answer = problems.get('answer', None)
    
    # 4개 선택지일 경우 '해당 없음' 추가 후 섞기
    if answer is not None:
        if len(choices) == 4:
            choices.append("해당 없음")
        
        # row id 기반 시드로 동일하게 섞기
        row_seed = hash(row['id']) % (2**32)
        local_random = random.Random(row_seed)
        
        indices = list(range(len(choices)))
        local_random.shuffle(indices)
        
        new_choices = [choices[i] for i in indices]
        original_answer_idx = answer - 1  # 0-indexed
        new_answer = indices.index(original_answer_idx) + 1  # 1-indexed
        
        choices = new_choices
        answer = new_answer
    
    # problems 컬럼으로 통합
    problems_dict = {
        'question': problems['question'],
        'choices': choices,
        'answer': answer
    }
    
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'problems': problems_dict,  # paragraph 다음 컬럼으로 위치시킬 수 있음
        'question_plus': problems.get('question_plus', None)
    }
    
    records.append(record)

# DataFrame 생성
df_new = pd.DataFrame(records)

# 컬럼 순서 재정렬: paragraph 다음에 problems 오도록
cols = df_new.columns.tolist()
cols.insert(cols.index('paragraph') + 1, cols.pop(cols.index('problems')))
df_new = df_new[cols]

# 확인
print(df_new.head())

# CSV 저장
df_new.to_csv('data/shuffled_dataset.csv', index=False, encoding='utf-8-sig')
print("Dataset saved to 'data/shuffled_dataset.csv'")
