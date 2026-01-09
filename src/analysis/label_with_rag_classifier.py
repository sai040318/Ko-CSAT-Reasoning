"""
HistoryClassifier(unsloth 모델)로 테스트 세트에 A/B 라벨을 붙이는 스크립트.

라벨 규칙 (rag_pipeline과 동일):
 A = 외부 문서 필요 (한국사 + paragraph에 근거 없음)
 B = 외부 문서 불필요 (비한국사이거나 paragraph에 근거 있음)
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# src/* 모듈 import를 위해 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import HistoryClassifier  # 기존 게이트 그대로 사용


def load_model(model_name: str, max_seq_length: int = 3072):
    """unsloth FastLanguageModel 로드."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def parse_problems(row):
    """row에서 question/choices/question_plus를 추출."""
    if "problems" in row and pd.notna(row["problems"]):
        prob = row["problems"]
        if isinstance(prob, str):
            try:
                prob = json.loads(prob)
            except Exception:
                try:
                    prob = ast.literal_eval(prob)
                except Exception:
                    prob = {}
        if isinstance(prob, dict):
            q = prob.get("question", "")
            c = prob.get("choices", [])
            qp = prob.get("question_plus", "")
            return q, c, qp
    # 플랫 컬럼 fallback
    return row.get("question", ""), row.get("choices", []), row.get("question_plus", "")


def main(args):
    model, tokenizer = load_model(args.model_name, args.max_seq_length)
    classifier = HistoryClassifier(model, tokenizer)

    df = pd.read_csv(args.input)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        paragraph = row.get("paragraph", "")
        question, choices, question_plus = parse_problems(row)
        # choices 문자열이면 리스트 변환 시도
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except Exception:
                try:
                    choices = ast.literal_eval(choices)
                except Exception:
                    choices = [choices]

        need_doc = classifier.is_external_doc_needed(paragraph, question, choices)
        label = "A" if need_doc else "B"
        results.append({"id": row.get("id", ""), "label": label})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"✅ saved: {args.output} (rows={len(out_df)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="src/data/test.csv")
    ap.add_argument("--output", default="gpt4_classification_results2.csv")
    ap.add_argument("--model-name", default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    ap.add_argument("--max-seq-length", type=int, default=3072)
    args = ap.parse_args()
    main(args)
