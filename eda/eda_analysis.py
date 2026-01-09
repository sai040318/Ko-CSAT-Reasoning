# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (eda_venv)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 수능형 문제 풀이 EDA 분석
#
# - **목표**: 수능형 문제(국어/사회) 정답 예측 (1~5 중 택1)
# - **평가**: Macro F1-score
# - **데이터**: Train 데이터 분석 (Test 데이터 분석 제외)

# %% [markdown]
# ## 0. 설정 및 라이브러리 임포트

# %%
# === 설정 ===
DATA_DIR = "/data/ephemeral/home/kdh/data"
QWEN3_MODEL = "Qwen/Qwen3-4B"  # 토크나이저용 (작은 모델로 토크나이저만 사용)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import ast
from collections import Counter
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일
sns.set_theme(style="whitegrid")

print("라이브러리 로드 완료")

# %% [markdown]
# ## 1. 데이터 로드 및 파싱

# %%
# 데이터 로드
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

print(f"Train shape: {train_df.shape}")

# %%
# problems 컬럼 파싱 함수
def parse_problems(problems_str: str) -> Dict:
    """problems 컬럼을 파싱하여 dict로 변환"""
    try:
        # ast.literal_eval 사용 (JSON보다 Python dict 형식에 적합)
        return ast.literal_eval(problems_str)
    except:
        try:
            return json.loads(problems_str.replace("'", '"'))
        except:
            return {'question': '', 'choices': [], 'answer': None}

# %%
# problems 파싱 및 컬럼 분리
def expand_problems(df: pd.DataFrame) -> pd.DataFrame:
    """problems 컬럼을 파싱하여 개별 컬럼으로 확장"""
    df = df.copy()

    # problems 파싱
    parsed = df['problems'].apply(parse_problems)

    # 개별 컬럼 추출
    df['question'] = parsed.apply(lambda x: x.get('question', ''))
    df['choices'] = parsed.apply(lambda x: x.get('choices', []))
    df['answer'] = parsed.apply(lambda x: x.get('answer', None))

    # 선택지 개수
    df['num_choices'] = df['choices'].apply(len)

    # question_plus 존재 여부
    df['has_question_plus'] = df['question_plus'].notna() & (df['question_plus'] != '')

    return df

# %%
# 데이터 확장
train_df = expand_problems(train_df)

print("=== Train 샘플 ===")
print(train_df[['id', 'question', 'num_choices', 'answer', 'has_question_plus']].head())

# %%
# 파싱 결과 확인
print("\n=== 파싱 결과 확인 ===")
print(f"Train - question 비어있는 행: {(train_df['question'] == '').sum()}")
print(f"Train - choices 비어있는 행: {(train_df['num_choices'] == 0).sum()}")

# %% [markdown]
# ## 2. 기본 정보

# %%
print("=== Train 기본 정보 ===")
print(train_df.info())
print("\n=== Null 값 확인 ===")
print(train_df.isnull().sum())

# %%
# 샘플 데이터 확인
print("\n=== Train 샘플 (첫 번째 행) ===")
sample = train_df.iloc[0]
print(f"ID: {sample['id']}")
print(f"\n[지문 (paragraph)]:\n{sample['paragraph'][:500]}...")
print(f"\n[질문 (question)]:\n{sample['question']}")
print(f"\n[선택지 (choices)]:\n{sample['choices']}")
print(f"\n[정답 (answer)]: {sample['answer']}")
print(f"\n[보기 (question_plus)]: {sample['question_plus'] if sample['has_question_plus'] else 'N/A'}")

# %% [markdown]
# ## 3. 텍스트 길이 분석

# %%
# 문자 수 기준 길이 계산
def calc_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """텍스트 길이 계산"""
    df = df.copy()

    # paragraph 길이
    df['paragraph_len'] = df['paragraph'].fillna('').apply(len)

    # question 길이
    df['question_len'] = df['question'].fillna('').apply(len)

    # choices 전체 길이 (모든 선택지 합)
    df['choices_len'] = df['choices'].apply(lambda x: sum(len(c) for c in x) if x else 0)

    # question_plus 길이
    df['question_plus_len'] = df['question_plus'].fillna('').apply(len)

    # 전체 입력 길이 추정
    df['total_len'] = df['paragraph_len'] + df['question_len'] + df['choices_len'] + df['question_plus_len']

    return df

train_df = calc_text_lengths(train_df)

# %%
# 길이 통계
print("=== Train 텍스트 길이 통계 (문자 수) ===")
length_cols = ['paragraph_len', 'question_len', 'choices_len', 'question_plus_len', 'total_len']
print(train_df[length_cols].describe())

# %%
# 길이 분포 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, col in enumerate(length_cols):
    ax = axes[idx // 3, idx % 3]

    # Train
    ax.hist(train_df[col], bins=50, alpha=0.7, color='blue')
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel('Length (chars)')
    ax.set_ylabel('Count')

# 마지막 subplot 숨기기
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/text_length_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 3.1 Qwen3 토큰 수 분석

# %%
# Qwen3 토크나이저 로드
from transformers import AutoTokenizer

print(f"Loading tokenizer: {QWEN3_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)
    print("토크나이저 로드 완료")
except Exception as e:
    print(f"토크나이저 로드 실패: {e}")
    print("대안으로 Qwen/Qwen2.5-1.5B 시도...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

# %%
# 토큰 수 계산 함수
def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 계산"""
    if not text or pd.isna(text):
        return 0
    return len(tokenizer.encode(str(text), add_special_tokens=False))

def count_tokens_batch(texts: List[str]) -> List[int]:
    """배치로 토큰 수 계산"""
    return [count_tokens(t) for t in texts]

# %%
# 토큰 수 계산 (시간이 걸릴 수 있음)
print("토큰 수 계산 중...")

train_df['paragraph_tokens'] = train_df['paragraph'].fillna('').apply(count_tokens)
train_df['question_tokens'] = train_df['question'].fillna('').apply(count_tokens)
train_df['choices_tokens'] = train_df['choices'].apply(lambda x: sum(count_tokens(c) for c in x) if x else 0)
train_df['question_plus_tokens'] = train_df['question_plus'].fillna('').apply(count_tokens)
train_df['total_tokens'] = train_df['paragraph_tokens'] + train_df['question_tokens'] + train_df['choices_tokens'] + train_df['question_plus_tokens']

print("토큰 수 계산 완료")

# %%
# 토큰 수 통계
print("=== Train 토큰 수 통계 (Qwen3) ===")
token_cols = ['paragraph_tokens', 'question_tokens', 'choices_tokens', 'question_plus_tokens', 'total_tokens']
print(train_df[token_cols].describe())

# %%
# 토큰 수 분포 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, col in enumerate(token_cols):
    ax = axes[idx // 3, idx % 3]

    ax.hist(train_df[col], bins=50, alpha=0.7, color='blue')
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Count')

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/token_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 긴 입력 분석 (Qwen3 context limit 고려)
print("\n=== 긴 입력 분석 ===")
for threshold in [1000, 2000, 4000, 8000]:
    train_count = (train_df['total_tokens'] > threshold).sum()
    print(f"총 토큰 > {threshold}: Train {train_count} ({train_count/len(train_df)*100:.1f}%)")

# %% [markdown]
# ## 4. 정답 분포 분석

# %%
# Train 정답 분포
print("=== Train 정답 분포 ===")
train_answer_dist = train_df['answer'].value_counts().sort_index()
print(train_answer_dist)
print(f"\n정답 비율:")
print((train_answer_dist / len(train_df) * 100).round(2))

# %%
# 정답 분포 시각화
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

colors = sns.color_palette("husl", 5)
bars = ax.bar(train_answer_dist.index.astype(str), train_answer_dist.values, color=colors)

ax.set_xlabel('Answer')
ax.set_ylabel('Count')
ax.set_title('Train Answer Distribution')

# 막대 위에 비율 표시
for bar, count in zip(bars, train_answer_dist.values):
    pct = count / len(train_df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/answer_distribution.png", dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## 5. 선택지 분석

# %%
# 선택지 개수 분포
print("=== 선택지 개수 분포 ===")
print("\n[Train]")
print(train_df['num_choices'].value_counts().sort_index())

# %%
# 선택지 개수 시각화
fig, ax = plt.subplots(figsize=(8, 5))

counts = train_df['num_choices'].value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values, color='steelblue')
ax.set_xlabel('Number of Choices')
ax.set_ylabel('Count')
ax.set_title('Train - Number of Choices Distribution')

for i, (idx, val) in enumerate(counts.items()):
    ax.text(i, val + 5, f'{val}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/num_choices_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 합답형 vs 일반형 분류
def classify_choice_type(choices: List[str]) -> str:
    """선택지 유형 분류"""
    if not choices:
        return 'empty'

    # 합답형 패턴: ㄱ, ㄴ, ㄷ, ㄹ 조합
    combo_pattern = r'^[ㄱㄴㄷㄹㅁ][,\s]*[ㄱㄴㄷㄹㅁ]'

    # 첫 번째 선택지로 판단
    first_choice = choices[0].strip()

    if re.search(combo_pattern, first_choice):
        return 'combination'  # 합답형 (ㄱ,ㄴ / ㄱ,ㄷ 등)
    elif first_choice.startswith('ㄱ') or first_choice.startswith('ㄴ'):
        return 'single_letter'  # 단일 ㄱ,ㄴ,ㄷ 형
    else:
        return 'general'  # 일반형

train_df['choice_type'] = train_df['choices'].apply(classify_choice_type)

# %%
# 선택지 유형 분포
print("=== 선택지 유형 분포 ===")
print("\n[Train]")
print(train_df['choice_type'].value_counts())

# %%
# 선택지 유형별 예시
print("\n=== 선택지 유형별 예시 ===")
for ctype in train_df['choice_type'].unique():
    sample = train_df[train_df['choice_type'] == ctype].iloc[0]
    print(f"\n[{ctype}]")
    print(f"Question: {sample['question'][:100]}...")
    print(f"Choices: {sample['choices']}")

# %%
# 선택지 유형별 정답 분포
print("\n=== 선택지 유형별 정답 분포 ===")
for ctype in train_df['choice_type'].unique():
    subset = train_df[train_df['choice_type'] == ctype]
    print(f"\n[{ctype}] (n={len(subset)})")
    print(subset['answer'].value_counts().sort_index())

# %%
# 시각화: 선택지 유형별 정답 분포
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, ctype in zip(axes, train_df['choice_type'].unique()):
    subset = train_df[train_df['choice_type'] == ctype]
    answer_dist = subset['answer'].value_counts().sort_index()

    ax.bar(answer_dist.index.astype(str), answer_dist.values, color='teal')
    ax.set_xlabel('Answer')
    ax.set_ylabel('Count')
    ax.set_title(f'{ctype} (n={len(subset)})')

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/choice_type_answer_dist.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. 문제 유형 분석 (부정어 포함)

# %%
# 부정어 패턴
NEGATIVE_PATTERNS = [
    r'않은\s*것',
    r'않는\s*것',
    r'아닌\s*것',
    r'없는\s*것',
    r'적절하지\s*않은',
    r'옳지\s*않은',
    r'일치하지\s*않는',
    r'거리가\s*먼',
    r'거리가\s*가장\s*먼',
]

def has_negative(question: str) -> bool:
    """질문에 부정어가 포함되어 있는지 확인"""
    if not question:
        return False
    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, question):
            return True
    return False

def extract_question_type(question: str) -> str:
    """질문 유형 추출"""
    if not question:
        return 'unknown'

    # 부정형
    if has_negative(question):
        return 'negative'

    # 긍정형 패턴
    if re.search(r'옳은\s*것|적절한\s*것|일치하는\s*것', question):
        return 'positive'

    # 추론형
    if re.search(r'추론|추리|유추|예상', question):
        return 'inference'

    # 이해형
    if re.search(r'이해|파악|설명', question):
        return 'comprehension'

    return 'other'

# %%
train_df['has_negative'] = train_df['question'].apply(has_negative)
train_df['question_type'] = train_df['question'].apply(extract_question_type)

# %%
# 부정어 포함 문제 분포
print("=== 부정어 포함 문제 분포 ===")
print("\n[Train]")
print(train_df['has_negative'].value_counts())
print(f"부정어 비율: {train_df['has_negative'].mean()*100:.1f}%")

# %%
# 문제 유형 분포
print("\n=== 문제 유형 분포 ===")
print("\n[Train]")
print(train_df['question_type'].value_counts())

# %%
# 부정어 문제 vs 긍정 문제 정답 분포 비교
print("\n=== 부정어 유무에 따른 정답 분포 ===")
print("\n[부정어 포함]")
print(train_df[train_df['has_negative']]['answer'].value_counts().sort_index())
print("\n[부정어 미포함]")
print(train_df[~train_df['has_negative']]['answer'].value_counts().sort_index())

# %%
# 시각화: 부정어 유무에 따른 정답 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (neg, title) in zip(axes, [(True, 'Negative Questions'), (False, 'Non-Negative Questions')]):
    subset = train_df[train_df['has_negative'] == neg]
    answer_dist = subset['answer'].value_counts().sort_index()

    ax.bar(answer_dist.index.astype(str), answer_dist.values, color='coral' if neg else 'skyblue')
    ax.set_xlabel('Answer')
    ax.set_ylabel('Count')
    ax.set_title(f'{title} (n={len(subset)})')

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/negative_answer_dist.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 부정어 패턴별 카운트
print("\n=== 부정어 패턴별 카운트 ===")
pattern_counts = {}
for pattern in NEGATIVE_PATTERNS:
    count = train_df['question'].apply(lambda x: bool(re.search(pattern, str(x)))).sum()
    if count > 0:
        pattern_counts[pattern] = count

for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    print(f"{pattern}: {count}")

# %% [markdown]
# ## 7. 키워드 기반 주제 분류

# %%
# 주제 키워드 정의
TOPIC_KEYWORDS = {
    'history_korea': ['조선', '고려', '신라', '백제', '고구려', '왕조', '임진왜란', '병자호란', '일제', '독립', '3.1운동', '광복'],
    'history_world': ['세계사', '로마', '그리스', '중국', '일본', '프랑스', '영국', '미국', '혁명', '전쟁', '제국'],
    'economics': ['경제', '시장', '가격', '수요', '공급', '인플레이션', 'GDP', '금리', '무역', '투자', '소비'],
    'politics': ['정치', '민주주의', '선거', '국회', '헌법', '정당', '권력', '국가', '정부', '법률'],
    'society': ['사회', '문화', '교육', '복지', '인권', '평등', '차별', '계층', '가족', '노동'],
    'korean_lang': ['독서', '글쓰기', '문학', '소설', '시', '수필', '비문학', '논설', '설명문', '작가'],
    'ethics': ['윤리', '도덕', '철학', '가치', '선', '악', '정의', '의무', '책임'],
}

def classify_topic(text: str) -> List[str]:
    """텍스트에서 주제 분류"""
    if not text:
        return ['unknown']

    topics = []
    text_lower = text.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                topics.append(topic)
                break

    return topics if topics else ['unknown']

# %%
# 주제 분류 적용
train_df['topics'] = train_df['paragraph'].apply(classify_topic)

# 첫 번째 주제만 추출 (대표 주제)
train_df['primary_topic'] = train_df['topics'].apply(lambda x: x[0])

# %%
# 주제 분포
print("=== 주제 분포 ===")
print("\n[Train]")
print(train_df['primary_topic'].value_counts())

# %%
# 시각화: 주제 분포
fig, ax = plt.subplots(figsize=(10, 6))

topic_counts = train_df['primary_topic'].value_counts()
ax.barh(topic_counts.index, topic_counts.values, color='mediumpurple')
ax.set_xlabel('Count')
ax.set_title('Train - Topic Distribution')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/topic_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 주제별 정답 분포
print("\n=== 주제별 정답 분포 ===")
for topic in train_df['primary_topic'].unique():
    subset = train_df[train_df['primary_topic'] == topic]
    if len(subset) >= 10:  # 최소 10개 이상인 주제만
        print(f"\n[{topic}] (n={len(subset)})")
        print(subset['answer'].value_counts().sort_index())

# %% [markdown]
# ### 7.1 TF-IDF 키워드 분석

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF 계산
tfidf = TfidfVectorizer(max_features=100, min_df=5, max_df=0.8)
train_tfidf = tfidf.fit_transform(train_df['paragraph'].fillna(''))

# 상위 키워드 추출
feature_names = tfidf.get_feature_names_out()
tfidf_scores = train_tfidf.sum(axis=0).A1
top_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: -x[1])[:30]

print("=== TF-IDF 상위 30 키워드 ===")
for keyword, score in top_keywords:
    print(f"{keyword}: {score:.2f}")

# %%
# TF-IDF 키워드 시각화
keywords, scores = zip(*top_keywords[:20])

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(keywords)), scores, color='steelblue')
ax.set_yticks(range(len(keywords)))
ax.set_yticklabels(keywords)
ax.set_xlabel('TF-IDF Score')
ax.set_title('Top 20 TF-IDF Keywords')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/tfidf_keywords.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. 텍스트 내용 분석 (형태소 분석)

# %%
# Kiwipiepy 로드
from kiwipiepy import Kiwi

kiwi = Kiwi()
print("Kiwipiepy 로드 완료")

# %%
# 형태소 분석 함수
def extract_nouns(text: str) -> List[str]:
    """텍스트에서 명사 추출"""
    if not text:
        return []

    result = kiwi.tokenize(text)
    # NNG(일반명사), NNP(고유명사) 추출
    nouns = [token.form for token in result if token.tag in ['NNG', 'NNP'] and len(token.form) > 1]
    return nouns

# %%
# 명사 추출 (샘플로 먼저 테스트)
print("명사 추출 테스트...")
sample_text = train_df['paragraph'].iloc[0]
sample_nouns = extract_nouns(sample_text)
print(f"샘플 명사: {sample_nouns[:20]}")

# %%
# 전체 명사 추출
print("\n전체 데이터 명사 추출 중...")
all_nouns = []
for text in train_df['paragraph'].fillna(''):
    all_nouns.extend(extract_nouns(text))

print(f"총 명사 수: {len(all_nouns)}")

# %%
# 명사 빈도 분석
noun_counter = Counter(all_nouns)
top_nouns = noun_counter.most_common(50)

print("\n=== 상위 50 명사 ===")
for noun, count in top_nouns:
    print(f"{noun}: {count}")

# %%
# WordCloud 생성
from wordcloud import WordCloud

# 한글 폰트 경로 (서버에 따라 다를 수 있음)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# 폰트 확인
import os
if not os.path.exists(font_path):
    font_path = None
    print("Warning: 한글 폰트를 찾을 수 없습니다. 기본 폰트 사용")

# WordCloud 생성
wc = WordCloud(
    font_path=font_path,
    width=800,
    height=400,
    background_color='white',
    max_words=100,
    colormap='viridis'
)

word_freq = dict(top_nouns)
wc.generate_from_frequencies(word_freq)

fig, ax = plt.subplots(figsize=(15, 8))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('Paragraph Word Cloud (Nouns)')

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/wordcloud.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 중복 지문 확인
print("\n=== 중복 지문 분석 ===")
paragraph_counts = train_df['paragraph'].value_counts()
duplicates = paragraph_counts[paragraph_counts > 1]

print(f"중복 지문 수: {len(duplicates)}")
print(f"중복 지문에 해당하는 문제 수: {duplicates.sum()}")

if len(duplicates) > 0:
    print("\n중복 지문 상위 5개:")
    for para, count in duplicates.head().items():
        print(f"- [{count}회] {para[:100]}...")

# %%
# 특수 마크업 패턴 분석
MARKUP_PATTERNS = {
    'circle_num': r'[㉠㉡㉢㉣㉤㉥㉦㉧㉨㉩]',  # 원문자
    'circle_alpha': r'[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙ]',  # 원알파벳
    'bracket_ga': r'\([가나다라마바]\)',  # (가), (나) 등
    'bracket_alpha': r'\[[A-Z]\]',  # [A], [B] 등
    'underline': r'㉠|㉡|_+[가-힣]+_+',  # 밑줄 표시
    'jamo': r'[ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ][\.,\s]',  # ㄱ. ㄴ. 형식
}

print("\n=== 특수 마크업 패턴 분석 ===")
for pattern_name, pattern in MARKUP_PATTERNS.items():
    train_count = train_df['paragraph'].apply(lambda x: bool(re.search(pattern, str(x)))).sum()
    print(f"{pattern_name}: Train {train_count}")

# %% [markdown]
# ## 10. 모델 입력 관점 분석 (Qwen3)

# %%
# 프롬프트 템플릿 시뮬레이션
PROMPT_TEMPLATE = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_TEMPLATE_WITH_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

def build_prompt(row: pd.Series) -> str:
    """프롬프트 생성"""
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(row['choices'])])

    if row['has_question_plus']:
        return PROMPT_TEMPLATE_WITH_PLUS.format(
            paragraph=row['paragraph'],
            question=row['question'],
            question_plus=row['question_plus'],
            choices=choices_str
        )
    else:
        return PROMPT_TEMPLATE.format(
            paragraph=row['paragraph'],
            question=row['question'],
            choices=choices_str
        )

# %%
# 전체 프롬프트 토큰 수 계산
print("전체 프롬프트 토큰 수 계산 중...")

train_df['full_prompt'] = train_df.apply(build_prompt, axis=1)
train_df['prompt_tokens'] = train_df['full_prompt'].apply(count_tokens)

print("계산 완료")

# %%
# 프롬프트 토큰 수 통계
print("=== 프롬프트 토큰 수 통계 (Qwen3) ===")
print("\n[Train]")
print(train_df['prompt_tokens'].describe())

# %%
# 프롬프트 토큰 수 분포 시각화
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(train_df['prompt_tokens'], bins=50, alpha=0.7, color='blue')
ax.axvline(x=2048, color='red', linestyle='--', label='2K limit')
ax.axvline(x=4096, color='green', linestyle='--', label='4K limit')
ax.axvline(x=8192, color='purple', linestyle='--', label='8K limit')
ax.set_xlabel('Prompt Tokens')
ax.set_ylabel('Count')
ax.set_title('Full Prompt Token Distribution (Qwen3)')
ax.legend()

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../eda/prompt_tokens_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# Context limit별 커버리지
print("\n=== Context Limit별 커버리지 ===")
for limit in [1024, 2048, 4096, 8192, 16384, 32768]:
    train_pct = (train_df['prompt_tokens'] <= limit).mean() * 100
    print(f"{limit:,} tokens: Train {train_pct:.1f}%")

# %%
# question_plus 유무별 분석
print("\n=== question_plus 유무별 프롬프트 토큰 수 ===")
print("\n[Train - question_plus 있음]")
print(train_df[train_df['has_question_plus']]['prompt_tokens'].describe())
print(f"개수: {train_df['has_question_plus'].sum()}")

print("\n[Train - question_plus 없음]")
print(train_df[~train_df['has_question_plus']]['prompt_tokens'].describe())
print(f"개수: {(~train_df['has_question_plus']).sum()}")

# %%
# 가장 긴 프롬프트 분석
print("\n=== 가장 긴 프롬프트 Top 5 ===")
top_long = train_df.nlargest(5, 'prompt_tokens')[['id', 'prompt_tokens', 'paragraph_len', 'has_question_plus']]
print(top_long)

# %% [markdown]
# ## 11. 분석 결과 요약

# %%
print("=" * 60)
print("EDA 분석 결과 요약")
print("=" * 60)

print(f"""
1. 데이터 크기
   - Train: {len(train_df):,}개

2. 텍스트 길이 (평균)
   - Paragraph: Train {train_df['paragraph_len'].mean():.0f}자
   - Question: Train {train_df['question_len'].mean():.0f}자
   - 전체 토큰: Train {train_df['total_tokens'].mean():.0f}

3. 정답 분포 (Train)
{train_df['answer'].value_counts().sort_index().to_string()}

4. 선택지
   - 4지선다: Train {(train_df['num_choices']==4).sum()}
   - 5지선다: Train {(train_df['num_choices']==5).sum()}

5. 문제 유형
   - 부정어 문제: Train {train_df['has_negative'].sum()} ({train_df['has_negative'].mean()*100:.1f}%)
   - 합답형: Train {(train_df['choice_type']=='combination').sum()}

6. 프롬프트 토큰 (Qwen3 기준)
   - 평균: Train {train_df['prompt_tokens'].mean():.0f}
   - 최대: Train {train_df['prompt_tokens'].max()}
   - 4K 이하: Train {(train_df['prompt_tokens']<=4096).mean()*100:.1f}%

7. question_plus 보기
   - 있음: Train {train_df['has_question_plus'].sum()}
""")

# %%
# 분석 결과 저장
summary_df = train_df[['id', 'num_choices', 'choice_type', 'has_negative', 'question_type',
                        'primary_topic', 'paragraph_len', 'total_tokens', 'prompt_tokens', 'has_question_plus', 'answer']]
summary_df.to_csv(f"{DATA_DIR}/../eda/train_analysis_summary.csv", index=False)
print(f"\n분석 결과 저장됨: {DATA_DIR}/../eda/train_analysis_summary.csv")

# %% [markdown]
# ---
# **EDA 분석 완료 (Train 데이터만)**
#
# 생성된 파일:
# - `text_length_distribution.png` - Train 텍스트 길이 분포
# - `token_distribution.png` - Train 토큰 수 분포
# - `answer_distribution.png` - Train 정답 분포
# - `num_choices_distribution.png` - Train 선택지 개수 분포
# - `choice_type_answer_dist.png` - 선택지 유형별 정답 분포
# - `negative_answer_dist.png` - 부정어 유무에 따른 정답 분포
# - `topic_distribution.png` - Train 주제 분포
# - `tfidf_keywords.png` - TF-IDF 키워드
# - `wordcloud.png` - 워드클라우드
# - `prompt_tokens_distribution.png` - Train 프롬프트 토큰 분포
# - `train_analysis_summary.csv` - Train 분석 요약
