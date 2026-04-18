import torch
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

from src.retrieval import WikipediaRetriever
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

    def is_korean_history(self, paragraph: str, question: str, choices: list | None) -> bool:
        choices_str = ""
        if choices:
            choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        prompt = f"""
        다음 문제는 한국사 문제인가?

        판단 기준:
        - 질문이 한국의 역사적 인물·사건·단체·제도·사상의 정체나 사실을 묻으면 한국사
        - 지문에 한국사 소재가 있어도, 지문 내용만 확인하는 독해 문제면 비한국사

        문제:
        {paragraph}

        질문:
        {question}

        선택지:
        {choices_str if choices_str else '없음'}

        A = 한국사
        B = 비한국사

        출력은 A 또는 B
        출력:"""
        return self._infer_AB(prompt) == "A"

    def need_external_doc(self, paragraph: str, question: str, choices: list | None) -> bool:
        choices_str = ""
        if choices:
            choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        prompt = f"""
        다음은 한국사 문제이다.

        당신의 임무는 paragraph 안에 정답을 선택할 수 있는 '근거 문장'이 존재하는지 여부를 판단하는 것이다.

        중요 규칙:
        - 추론 금지
        - 상식 사용 금지
        - 의미 유추 금지
        - paragraph에 없는 정보를 알고 있어야만 정답을 고를 수 있다면 외부 문서가 필요하다고 판단한다.

        판단:
        A = paragraph에 정답의 근거가 없음 (외부 문서 필요)
        B = paragraph에 정답의 근거가 있음 (외부 문서 불필요)

        paragraph:
        {paragraph}

        question:
        {question}

        choices:
        {choices_str if choices_str else '없음'}

        출력은 반드시 A 또는 B 한 글자.
        출력:"""
        return self._infer_AB(prompt) == "A"

    def is_external_doc_needed(self, paragraph: str, question: str, choices: list | None) -> bool:
        if not self.is_korean_history(paragraph, question, choices):
            return False

        return self.need_external_doc(paragraph, question, choices)


# =====================================================
# RAGPipeline (Wikipedia)
# =====================================================
class RAGPipeline:
    def __init__(
        self,
        corpus_path: str,
        top_k: int = 3,
        retriever: WikipediaRetriever | None = None,
    ):
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.retriever = retriever

    def retrieve_from_corpus(self, paragraph: str, question: str, choices: list) -> str:
        """
        지문+질문으로 Wikipedia 검색을 수행하고, 구조화된 컨텍스트 문자열을 반환.
        """
        if self.retriever is None:
            raise ValueError("RAGPipeline.retriever가 설정되어 있지 않습니다.")

        query = f"{paragraph}\n{question}".strip()
        results = self.retriever.retrieve(query, top_k=self.top_k)

        contexts = []
        for idx, r in enumerate(results, start=1):
            title = r.get("title", "제목 없음")
            meta = r.get("metadata") or {}
            content = meta.get("full_content") or r.get("content", "")

            formatted_doc = (
                f"#### {idx}. {title}\n"
                f"{content}"
            )
            contexts.append(formatted_doc)

        preface = (
            f"아래는 문제 해결을 위해 검색된 위키피디아 배경지식(참고 문서 {len(contexts)}건)입니다. "
            "이 문서들에는 정답과 관련된 핵심 정보뿐만 아니라 관련 없는 내용(노이즈)도 섞여 있을 수 있습니다. "
            "반드시 **문맥에 맞는 정보만 선별**하여 정답을 추론하세요.\n\n"
        )

        full_context = "\n\n---\n\n".join(contexts)

        return preface + full_context

    def add_documents_to_df(self, df: pd.DataFrame, history_classifier: HistoryClassifier) -> pd.DataFrame:
        documents = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Wikipedia RAG 문서 생성"):
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

    data_path = "src/data/train.csv"
    df = pd.read_csv(data_path)
    retriever = WikipediaRetriever(top_k=5)
    rag = RAGPipeline(corpus_path="src/corpus/corpus.json", retriever=retriever)
    df_with_docs = rag.add_documents_to_df(df, history_classifier)

    out_csv_path = "data/self_instruct_with_wiki_documents.csv"
    df_with_docs.to_csv(out_csv_path, index=False)

    print(f"✅ CSV 저장 완료: {out_csv_path}")
