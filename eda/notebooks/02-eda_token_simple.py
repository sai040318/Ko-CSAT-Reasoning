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

# %% id="my6SEWO336qj"
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import warnings


# %% id="RHWr7XFf3k2X"
# Load the train dataset
# TODO Train Data 경로 입력
dataset = pd.read_csv('train.csv')

# Flatten the JSON dataset
records = []
for _, row in dataset.iterrows():
    problems = literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
        "question_plus": problems.get('question_plus', None),
    }
    # Include 'question_plus' if it exists
    if 'question_plus' in problems:
        record['question_plus'] = problems['question_plus']
    records.append(record)

# Convert to DataFrame
df = pd.DataFrame(records)

# %% [markdown] id="-26d9ZOHjLkJ"
# ### Token

# %% colab={"base_uri": "https://localhost:8080/"} id="WLaKCegLiPX8" outputId="e41594f8-e5b7-44a8-e224-e6200ca8a4f8"
# Tokenizer 로드
print("Tokenizer 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-2b")
print("✓ Tokenizer 로드 완료!")

# %% colab={"base_uri": "https://localhost:8080/"} id="mfQ82zSOiSG_" outputId="bd7d7ca3-033f-4dae-f44d-9d9daedfc343"
print("Paragraph Token 분석 시작")
print("=" * 50)

# Token 수 계산
print("\nToken 수 계산 중... (시간이 걸릴 수 있습니다)")
df['token_count'] = df['paragraph'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
print("✓ Token 수 계산 완료!")

# %% colab={"base_uri": "https://localhost:8080/"} id="HSGY-XeViXBj" outputId="2968ec05-4b6a-4a70-f04c-1f72a70224e6"
# ===== 기본 통계 =====
print("\n" + "=" * 50)
print("1. 기본 통계")
print("=" * 50)
print(f"전체 데이터 개수: {len(df)}")
print(f"평균 Token 수: {df['token_count'].mean():.2f}")
print(f"중앙값 Token 수: {df['token_count'].median():.2f}")
print(f"표준편차: {df['token_count'].std():.2f}")
print(f"최소 Token 수: {df['token_count'].min()}")
print(f"최대 Token 수: {df['token_count'].max()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="CDPtM1CcjSmw" outputId="ea9aa545-8f87-4658-ba02-ab28d10e1c9a"
# ===== Token 범위별 분포 =====
print("\n" + "=" * 50)
print("3. Token 범위별 분포")
print("=" * 50)
bins = [0, 100, 200, 300, 400, 500, 1000, 2000, float('inf')]
labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-1000', '1000-2000', '2000+']
df['token_range'] = pd.cut(df['token_count'], bins=bins, labels=labels)

range_counts = df['token_range'].value_counts().sort_index()
for range_label, count in range_counts.items():
    pct = count / len(df) * 100
    print(f"{range_label} tokens: {count}개 ({pct:.2f}%)")

# %% id="e0jIQzLIgecy"
