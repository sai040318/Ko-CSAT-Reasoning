# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from ast import literal_eval
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# Load dataset
dataset = pd.read_csv('train.csv')

# Flatten JSON
records = []
for _, row in dataset.iterrows():
    problems = literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
        'question_plus': problems.get('question_plus', None),
    }
    records.append(record)

df = pd.DataFrame(records)

# %%
df = pd.DataFrame(records)

df.info()
print("📊 기본 데이터 정보")
print("="*80)
print(f"전체 샘플 수: {len(df)}")
print(f"컬럼: {list(df.columns)}")
print(f"\n결측치:\n{df.isnull().sum()}")
print(f"\n데이터 타입:\n{df.dtypes}")

df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="Y95lFBOE23fX" outputId="5d257a4c-c243-4fc6-e7e8-1b21dfb4ba4c"
import matplotlib.pyplot as plt
import numpy as np

answer_counts = df['answer'].value_counts(dropna=True)
answer_ratios = df['answer'].value_counts(normalize=True, dropna=True) * 100

print(answer_counts)
print(answer_ratios)

plt.figure(figsize=(6,4))
answer_counts.plot(kind='bar')
plt.title("Answer Distribution")
plt.xlabel("Answer")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


# %% id="e0jIQzLIgecy"
df['question_plus'] = df['question_plus'].fillna('')
df['full_question'] = df.apply(
    lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'],
    axis=1
)

df['question_length'] = df['full_question'].apply(len)

print("\n[Question 통계]")
print(f"평균: {df['question_length'].mean():.2f}")
print(f"중앙값: {df['question_length'].median():.2f}")
print(f"최소: {df['question_length'].min()}")
print(f"최대: {df['question_length'].max()}")
print(f"표준편차: {df['question_length'].std():.2f}")

plt.figure(figsize=(5,3))
plt.hist(df['question_length'], bins=30, edgecolor='black')
plt.title('Distribution of Question Lengths')
plt.xlabel('Question Length')
plt.ylabel('Frequency')
plt.show()


# %%

# %%
df['paragraph_length'] = df['paragraph'].apply(len)

print("\n[Paragraph 통계]")
print(f"평균: {df['paragraph_length'].mean():.2f}")
print(f"중앙값: {df['paragraph_length'].median():.2f}")
print(f"최소: {df['paragraph_length'].min()}")
print(f"최대: {df['paragraph_length'].max()}")
print(f"표준편차: {df['paragraph_length'].std():.2f}")

plt.figure(figsize=(5,3))
plt.hist(df['paragraph_length'], bins=30, edgecolor='black')
plt.title('Distribution of Paragraph Lengths')
plt.xlabel('Paragraph Length')
plt.ylabel('Frequency')
plt.show()


# %%
# 100자 미만 필터
df_short = df[df['paragraph_length'] < 100]

# 100자 미만 비율
ratio = df_short.shape[0] / df.shape[0] * 100

# 랜덤 샘플 10개 (데이터가 10개 미만이면 전부)
sample_records = df_short.sample(n=min(10, len(df_short)), random_state=42)
pd.set_option('display.max_colwidth', None)
print(f"100자 미만 paragraph 갯수: {df_short.shape[0]}")
print(f"100자 미만 paragraph 비율: {ratio:.2f}%\n")
print("샘플 10개:")
display(sample_records[['paragraph','paragraph_length']].reset_index().set_index('index'))

# %%
df_underlined = df[df['question'].str.contains('밑줄 친')]

num_samples = len(df_underlined)
print(f'Number of samples with "밑줄 친": {num_samples}')

plt.figure(figsize=(8,5))
plt.hist(df_underlined['paragraph_length'], bins=10, color='skyblue', edgecolor='black')
plt.title('Paragraph Length Distribution for Questions with "underlined"')
plt.xlabel('Paragraph Length (characters)')
plt.ylabel('Sample Count')
plt.show()

# %%
df_filtered = df[~df['question'].str.contains('밑줄 친') & (df['paragraph_length'] < 100)]

display(df_filtered[['paragraph','paragraph_length']].reset_index())
