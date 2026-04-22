import unsloth
import hydra
import pandas as pd
import os
import sys
import re
from pathlib import Path
from ast import literal_eval

import wandb

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from src.utils.utils import set_seed


import src.model  
import src.data  
from src.retrieval import EnsembleRetriever, WikipediaRetriever


def get_rag_components(cfg: DictConfig):
    retriever_name = cfg.get("rag", {}).get("retriever", "").lower()

    if retriever_name == "wikipedia":
        from src.rag.rag_wiki_pipeline import RAGPipeline, HistoryClassifier
    else:
        from src.rag.rag_pipeline import RAGPipeline, HistoryClassifier

    return RAGPipeline, HistoryClassifier

# Hydra를 통해 설정 파일을 로드합니다.
@hydra.main(version_base=None, config_path=str(project_root / "config"), config_name="config")
def main(cfg: DictConfig):
    # 난수 시드 고정
    set_seed(cfg.seed)
    
    # print(OmegaConf.to_yaml(cfg))

    # 모델 클래스 로드 및 Tokenizer 초기화
    model_cls = MODEL_REGISTRY.get(cfg.model.type)
    
    if cfg.mode == "train":
        print("🚀 학습 모드 시작")
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            max_seq_length=cfg.model.max_seq_length,
            lora_bias=cfg.model.lora_bias,
            **cfg.training 
        )

        tokenizer = model.tokenizer

        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)

        # =================================================================
        # RAG 적용 (train 모드)
        # =================================================================
        if cfg.get("rag", {}).get("use", False):
            print("[Train Mode] RAG Pipeline 활성화: 학습 데이터에 문서를 검색하여 주입합니다.")
            from datasets import Dataset as HFDataset

            if getattr(dataset, "dataset", None) is None:
                dataset.load_data()

            RAGPipeline, HistoryClassifier = get_rag_components(cfg)
            history_classifier = HistoryClassifier(model.model, tokenizer)
            if cfg.rag.get("retriever", "").lower() == "wikipedia":
                retriever = WikipediaRetriever(
                    lang=cfg.rag.get("wiki_lang", "ko"),
                    top_k=cfg.rag.get("top_k", 5),
                )
            else:
                retriever = EnsembleRetriever(
                    corpus_path="src/corpus/corpus.json",
                    bm25_k=cfg.rag.get("bm25_k", 10),
                    vec_k=cfg.rag.get("vec_k", 10),
                    top_k=cfg.rag.get("top_k", 5),
                    weight_bm25=cfg.rag.get("weight_bm25", 0.5),
                    weight_vec=cfg.rag.get("weight_vec", 0.5),
                )
            rag_pipeline = RAGPipeline(
                corpus_path="src/corpus/corpus.json",
                top_k=cfg.rag.get("top_k", 5),
                retriever=retriever,
            )

            # BaselineDataset 구조상 self.dataset["train"]에 데이터가 있음
            # 원본 CSV 사용 (problems 컬럼 유지)
            df_raw = pd.read_csv(cfg.dataset.path)
            df_with_docs = rag_pipeline.add_documents_to_df(df_raw, history_classifier)
            df_with_docs["paragraph"] = df_with_docs.apply(
                lambda r: f"### 참고 문서 (Background Knowledge)\n{r['documents']}\n\n---\n### 문제 지문\n{r['paragraph']}"
                if pd.notna(r["documents"]) and r["documents"]
                else r["paragraph"],
                axis=1,
            )
            # problems를 풀어 평탄화
            records = []
            for _, row in df_with_docs.iterrows():
                problems = literal_eval(row["problems"]) if isinstance(row["problems"], str) else row["problems"]
                records.append(
                    {
                        "id": row["id"],
                        "paragraph": row["paragraph"],
                        "question": problems.get("question", ""),
                        "choices": problems.get("choices", []),
                        "answer": int(problems["answer"]) if problems.get("answer") is not None else None,
                        "question_plus": problems.get("question_plus", None),
                        "documents": None,  # paragraph에 이미 삽입됨 — 외부문서 이중 주입 방지
                    }
                )
            dataset.dataset["train"] = HFDataset.from_pandas(pd.DataFrame(records))
            # 캐시 저장
            augmented_out = cfg.rag.get("augmented_train_path", "output/train_with_context_cached.csv")
            os.makedirs(os.path.dirname(augmented_out), exist_ok=True)
            pd.DataFrame(records).to_csv(augmented_out, index=False)
            print("학습 데이터 RAG 주입 완료")
        # =================================================================

        # Dataset 로드 및 전처리
        processed_dataset = dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **cfg.dataset.preprocess.train
        )
        from datasets import DatasetDict
        
        full_dataset = processed_dataset["train"]
        # dataset split
        split_ratio = cfg.dataset.get("split_ratio", 0.8)  

        split_dataset = full_dataset.train_test_split(
            train_size=split_ratio,
            seed=cfg.seed
        )
        
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        model.train(
            train_dataset=train_dataset,
            eval_dataset=None,
            **cfg.training
        )
        
    elif cfg.mode == "inference":
        print("🚀 추론 모드 시작")
        
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            skip_init=True,  
            max_seq_length=cfg.model.max_seq_length,
        )
        tokenizer = model.tokenizer
        
        model_load_path = cfg.inference.get("model_load_path", cfg.training.output_dir)
        if not os.path.exists(model_load_path):
            raise ValueError(f"모델 경로를 찾을 수 없습니다: {model_load_path}")
        # adapter_config.json 있는 checkpoint 중 마지막 것을 우선 사용
        # Unsloth checkpoint-N/에 adapter_config.json이 없으면 save_model()로 저장된 루트 경로 사용
        p = Path(model_load_path)
        ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir() and (d / "adapter_config.json").exists()]
        if ckpts:
            ckpts.sort(
                key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
            )
            model_load_path = str(ckpts[-1])

        print(f"모델 로드 중: {model_load_path}")
        model.load_model(model_load_path)
        
        test_dataset_path = cfg.inference.get("test_dataset_path", "data/test.csv")
        
        print(f"테스트 데이터셋 로드 중: {test_dataset_path}")
        test_dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        test_dataset = test_dataset_cls(test_dataset_path)

        ### RAG부분 ###
        RAGPipeline, HistoryClassifier = get_rag_components(cfg)
        history_classifier = HistoryClassifier(model.model, tokenizer)
        # rag 설정이 있고 use가 True일 때만 retriever 주입
        if cfg.get("rag", {}).get("use", False):
            from datasets import Dataset as HFDataset, DatasetDict

            if getattr(test_dataset, "dataset", None) is None:
                test_dataset.load_data()

            if cfg.rag.get("retriever", "").lower() == "wikipedia":
                retriever = WikipediaRetriever(
                    lang=cfg.rag.get("wiki_lang", "ko"),
                    top_k=cfg.rag.get("top_k", 5),
                )
            else:
                retriever = EnsembleRetriever(
                    corpus_path="src/corpus/corpus.json",
                    bm25_k=cfg.rag.get("bm25_k", 10),
                    vec_k=cfg.rag.get("vec_k", 10),
                    top_k=cfg.rag.get("top_k", 5),
                    weight_bm25=cfg.rag.get("weight_bm25", 0.5),
                    weight_vec=cfg.rag.get("weight_vec", 0.5),
                )
            rag_pipeline = RAGPipeline(
                corpus_path="src/corpus/corpus.json",
                top_k=cfg.rag.get("top_k", 5),
                retriever=retriever,
            )

            
            # 원본 CSV 사용 (problems 컬럼 유지)
            df_raw = pd.read_csv(test_dataset_path)
            df_with_docs = rag_pipeline.add_documents_to_df(df_raw, history_classifier)
            df_with_docs["paragraph"] = df_with_docs.apply(
                lambda r: f"### 참고 문서 (Background Knowledge)\n{r['documents']}\n\n---\n### 문제 지문\n{r['paragraph']}"
                if pd.notna(r["documents"]) and r["documents"]
                else r["paragraph"],
                axis=1,
            )
            # problems를 풀어 평탄화
            records = []
            for _, row in df_with_docs.iterrows():
                problems = literal_eval(row["problems"]) if isinstance(row["problems"], str) else row["problems"]
                records.append(
                    {
                        "id": row["id"],
                        "paragraph": row["paragraph"],
                        "question": problems.get("question", ""),
                        "choices": problems.get("choices", []),
                        "answer": int(problems["answer"]) if problems.get("answer") is not None else None,
                        "question_plus": problems.get("question_plus", None),
                        "documents": None,  # paragraph에 이미 삽입됨 — 외부문서 이중 주입 방지
                    }
                )
            ds = HFDataset.from_pandas(pd.DataFrame(records))
            test_dataset.dataset = DatasetDict({"train": ds})
        else:
            rag_pipeline = None

        test_dataset.extra_columns["mode"] = cfg.mode
        test_dataset.extra_columns["history_classifier"] = history_classifier
        test_dataset.extra_columns["rag_pipeline"] = rag_pipeline

        processed_test_dataset = test_dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **cfg.dataset.preprocess.inference
        )

        predictions = model.predict(
            dataset=processed_test_dataset["train"],
            **cfg.inference
        )
        

        output_path = cfg.inference.get("output_path", "output/output.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_output = pd.DataFrame([
            {"id": id, "answer": answer} 
            for id, answer in predictions.items()
        ])
        df_output = df_output.sort_values("id")
        df_output.to_csv(output_path, index=False)
        print(f"결과 저장 완료: {output_path}")
        
    elif cfg.mode == "evaluate":
        print("🚀 평가 모드 시작")

        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            skip_init=True, 
            max_seq_length=cfg.model.max_seq_length,
        )

        tokenizer = model.tokenizer

        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)
        
        # =================================================================
        # RAG 적용 (evaluate 모드)
        # =================================================================
        if cfg.get("rag", {}).get("use", False):
            print("🚀 [Eval Mode] RAG Pipeline 활성화: 평가 데이터에 문서를 검색하여 주입합니다.")
            from datasets import Dataset as HFDataset

            if getattr(dataset, "dataset", None) is None:
                dataset.load_data()

            RAGPipeline, HistoryClassifier = get_rag_components(cfg)
            history_classifier = HistoryClassifier(model.model, tokenizer)
            if cfg.rag.get("retriever", "").lower() == "wikipedia":
                retriever = WikipediaRetriever(
                    lang=cfg.rag.get("wiki_lang", "ko"),
                    top_k=cfg.rag.get("top_k", 5),
                )
            else:
                retriever = EnsembleRetriever(
                    corpus_path="src/corpus/corpus.json",
                    bm25_k=cfg.rag.get("bm25_k", 10),
                    vec_k=cfg.rag.get("vec_k", 10),
                    top_k=cfg.rag.get("top_k", 5),
                    weight_bm25=cfg.rag.get("weight_bm25", 0.5),
                    weight_vec=cfg.rag.get("weight_vec", 0.5),
                )
            rag_pipeline = RAGPipeline(
                corpus_path="src/corpus/corpus.json",
                top_k=cfg.rag.get("top_k", 5),
                retriever=retriever,
            )

            # 원본 CSV 사용 (problems 컬럼 유지)
            df_raw = pd.read_csv(cfg.dataset.path)
            df_with_docs = rag_pipeline.add_documents_to_df(df_raw, history_classifier)
            df_with_docs["paragraph"] = df_with_docs.apply(
                lambda r: f"### 참고 문서 (Background Knowledge)\n{r['documents']}\n\n---\n### 문제 지문\n{r['paragraph']}"
                if pd.notna(r["documents"]) and r["documents"]
                else r["paragraph"],
                axis=1,
            )
            # problems를 풀어 평탄화
            records = []
            for _, row in df_with_docs.iterrows():
                problems = literal_eval(row["problems"]) if isinstance(row["problems"], str) else row["problems"]
                records.append(
                    {
                        "id": row["id"],
                        "paragraph": row["paragraph"],
                        "question": problems.get("question", ""),
                        "choices": problems.get("choices", []),
                        "answer": int(problems["answer"]) if problems.get("answer") is not None else None,
                        "question_plus": problems.get("question_plus", None),
                        "documents": None,  # paragraph에 이미 삽입됨 — 외부문서 이중 주입 방지
                    }
                )
            dataset.dataset["train"] = HFDataset.from_pandas(pd.DataFrame(records))
            print("평가 데이터 RAG 주입 완료")
        # =================================================================
        eval_preprocess_config = cfg.dataset.preprocess.get("evaluate", cfg.dataset.preprocess.inference)
        # Dataset 로드 및 전처리
        processed_dataset = dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **eval_preprocess_config  
        )
        # 평가 모드에서는 입력된 평가용 CSV 전체를 그대로 사용합니다.
        eval_dataset = processed_dataset["train"]

        model_load_path = cfg.evaluate.get("model_load_path", cfg.training.output_dir)

        # adapter_config.json 있는 checkpoint 중 마지막 것을 우선 사용
        # Unsloth checkpoint-N/에 adapter_config.json이 없으면 save_model()로 저장된 루트 경로 사용
        p = Path(model_load_path)
        ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir() and (d / "adapter_config.json").exists()]
        if ckpts:
            ckpts.sort(
                key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
            )
            model_load_path = str(ckpts[-1])
        
        print(f"모델 로드 중: {model_load_path}")
        model.load_model(model_load_path)
        
        metrics = model.evaluate(
            eval_dataset,
            original_dataset_path=cfg.dataset.path,
            eval_output_path=cfg.evaluate.get("eval_output_path", "output/eval_results.csv")
        )
        print(f"평가 결과 : {metrics}")

        rag_cfg = cfg.get("rag", {})
        wandb.init(
            project=cfg.get("wandb_project", "ko-csat-reasoning"),
            name=cfg.get("exp_name", "eval"),
            config={
                "model": cfg.model.model_name_or_path,
                "rag_use": rag_cfg.get("use", False),
                "retriever": rag_cfg.get("retriever", "none"),
                "top_k": rag_cfg.get("top_k", 0),
                "weight_bm25": rag_cfg.get("weight_bm25", None),
                "weight_vec": rag_cfg.get("weight_vec", None),
                "dataset": cfg.dataset.path,
            },
            reinit=True,
        )
        wandb.log({
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "correct": metrics["correct"],
            "total": metrics["total"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
