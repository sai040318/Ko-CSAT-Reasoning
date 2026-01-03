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
#코드는 그대로 실행해주시면 됩니다.

# %%
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# %%
# 경로 맞게 수정
df_raw = pd.read_csv("train.csv")

print("전체 row 수:", len(df_raw))
df_raw.head()


# %%
records = []

for _, row in df_raw.iterrows():
    p = literal_eval(row["problems"])
    records.append({
        "id": row["id"],
        "paragraph": row["paragraph"],
        "question": p.get("question"),
        "question_plus": p.get("question_plus"),
        "choices": p.get("choices"),
        "answer" : p.get("answer"),
    })

df = pd.DataFrame(records)
df.head()



# %%
if "external_knowledge" not in df.columns:
    df["external_knowledge"] = pd.NA

print(df["external_knowledge"].isna().sum())


# %%
# 아직 라벨링 안 된 첫 index
current_idx = df[df["external_knowledge"].isna()].index.min()
current_idx


# %%
def show_problem(i):
    row = df.loc[i]
    print("=" * 80)
    print(f"[INDEX] {i}")
    print(f"[ID] {row['id']}\n")
    
    print("📘 Paragraph")
    print(row["paragraph"], "\n")
    
    print("❓ Question")
    q = row["question"]
    if pd.notna(row["question_plus"]):
        q += " " + row["question_plus"]
    print(q, "\n")
    
    print("🔢 Choices")
    for c in row["choices"]:
        print("-", c)
    print("answer")
    print(row["answer"])
        
    print("\n[입력 가이드]")
    print("0 = 지문만으로 가능")
    print("1 = 외부 지식 필요")



# %%
i = current_idx  #이것 시작 숫자만 바꾸면됨 다형님 1000-1199, 민석님 1200-1399 준범님 1400-1599 태원님 1600 - 1799, 승환님 1800 - 2031

show_problem(i)


# %%
# 👉 이셀만 실행하면됩니다. 
df.at[i, "external_knowledge"] = 0 # 내부추론만 실행해되는거면 0 , 외부지식 필요한거같은거 1 
i += 1
show_problem(i)

# %%
total = len(df)
done = df["external_knowledge"].notna().sum()

print(f"진행률: {done}/{total} ({done/total*100:.2f}%)")

# %%
df.to_csv("train_labeled_partial.csv", index=False)
print("💾 중간 저장 완료")

