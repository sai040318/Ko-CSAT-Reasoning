import torch
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from unsloth import FastLanguageModel

# =====================================================
# HistoryClassifier (2-stage LLM Gate)
# =====================================================
class HistoryClassifier:
    """
    1차: 한국사 여부 판정
    2차: (한국사일 때만) paragraph 내 선택지 명시 포함 여부 판정

    return True  -> 외부문서 필요 (A)
    return False -> 외부문서 불필요 (B)
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.id_A = tokenizer.encode("A", add_special_tokens=False)[0]
        self.id_B = tokenizer.encode("B", add_special_tokens=False)[0]

    # -----------------------------
    # 공통 A/B inference
    # -----------------------------
    @torch.no_grad()
    def _infer_AB(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self.model.device)

        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]

        return "A" if logits[self.id_A] > logits[self.id_B] else "B"

    # -----------------------------
    # 1차: 한국사 여부
    # -----------------------------
    def is_korean_history(self, paragraph: str, question: str, choices: list | None) -> bool:
        choices_str = ""
        if choices:
            choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

        prompt = f"""
        다음 문제는 한국사 문제인가?

        판단 기준:
        - 한국의 역사적 사건, 인물, 제도, 시기 → 한국사
        - 그 외 세계사, 과학, 일반상식 등 → 비한국사

        문제:
        {paragraph}

        질문:
        {question}

        선택지:
        {choices_str if choices_str else '없음'}

        판단:
        A = 한국사
        B = 비한국사

        출력은 반드시 A 또는 B 한 글자.
        출력:"""
        return self._infer_AB(prompt) == "A"

    # -----------------------------
    # 2차: paragraph 내 명시 포함 여부
    # -----------------------------
    def need_external_doc(self, paragraph: str, question: str, choices: list | None) -> bool:
        choices_str = ""
        if choices:
            choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

        prompt = f"""
        다음은 한국사 문제이다.

        당신의 임무는 선택지 중 하나가 paragraph 안에
        직접적으로 등장하는지 여부만 판단하는 것이다.

        중요 규칙:
        - 추론 금지
        - 상식 사용 금지
        - 의미 유추 금지
        - 실제로 등장한 표현만 인정

        판단:
        A = 선택지 중 어느 것도 paragraph에 명시적으로 등장하지 않음
        B = 선택지 중 하나 이상이 paragraph에 명시적으로 등장함

        paragraph:
        {paragraph}

        question:
        {question}

        choices:
        {choices_str if choices_str else '없음'}

        출력은 반드시 A 또는 B 한 글자.
        출력:"""
        # A → 외부문서 필요
        return self._infer_AB(prompt) == "A"

    # -----------------------------
    # 최종 gate
    # -----------------------------
    def is_external_doc_needed(self, paragraph: str, question: str, choices: list | None) -> bool:
        # 비한국사는 무조건 외부문서 불필요
        if not self.is_korean_history(paragraph, question, choices):
            return False

        # 한국사일 때만 paragraph 검사
        return self.need_external_doc(paragraph, question, choices)


# =====================================================
# RAGPipeline
# =====================================================
class RAGPipeline:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path

    def retrieve_from_corpus(self, paragraph: str, question: str, choices: list) -> str:
        # 실제 검색 로직으로 교체
        return "임시 문서"

    def add_documents_to_df(self, df: pd.DataFrame, history_classifier: HistoryClassifier) -> pd.DataFrame:
        documents = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="RAG 문서 생성"):
            problems = (
                literal_eval(row["problems"]) if isinstance(row["problems"], str) else row["problems"]
            )

            paragraph = row.get("paragraph", "")
            question = problems.get("question", "")
            choices = problems.get("choices", [])

            if history_classifier.is_external_doc_needed(paragraph, question, choices):
                doc = self.retrieve_from_corpus(paragraph, question, choices)
            else:
                doc = None

            documents.append(doc)

        df_out = df.copy()
        df_out["documents"] = documents
        return df_out


if __name__ == "__main__":
    print("✅ Unsloth 모델 로딩 중...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        max_seq_length=3072,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    FastLanguageModel.for_inference(model)

    history_classifier = HistoryClassifier(model, tokenizer)

    data_path = "data/train.csv"
    df = pd.read_csv(data_path)

    rag = RAGPipeline(corpus_path="./corpus")
    df_with_docs = rag.add_documents_to_df(df, history_classifier)

    out_csv_path = "data/self_instruct_with_documents22.csv"
    df_with_docs.to_csv(out_csv_path, index=False)

    print(f"✅ CSV 저장 완료: {out_csv_path}")
