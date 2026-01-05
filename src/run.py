import unsloth 
import hydra
import pandas as pd
import os
import sys
import re
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from src.utils.utils import set_seed

# 레지스트리에 모델과 데이터셋을 등록하기 위해 import
# __init__.py에서 자동으로 baseline_model과 baseline_data를 import함
import src.model  # noqa: F401
import src.data  # noqa: F401 

from src.rag.rag_pipeline import RAGPipeline, HistoryClassifier
from src.retrieval import EnsembleRetriever

# Hydra를 통해 설정 파일을 로드합니다.
# config_path는 프로젝트 루트 기준으로 설정
@hydra.main(version_base=None, config_path=str(project_root / "config"), config_name="config")
def main(cfg: DictConfig):
    # 난수 시드 고정
    set_seed(cfg.seed)
    
    print(OmegaConf.to_yaml(cfg))

    # 모델 클래스 로드 및 Tokenizer 초기화
    # 모델에 맞는 토크나이저(Chat Template 포함)를 가져오기 위해 모델 클래스를 먼저 로드합니다.
    model_cls = MODEL_REGISTRY.get(cfg.model.type)
    
    # 실행 모드에 따른 동작 수행
    if cfg.mode == "train":

        # Model 초기화
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            max_seq_length=cfg.model.max_seq_length,
            lora_bias=cfg.model.lora_bias,
            **cfg.training # 학습 관련 설정 전달
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

            history_classifier = HistoryClassifier(model.model, tokenizer)
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
            df_raw = dataset.dataset["train"].to_pandas()
            df_with_docs = rag_pipeline.add_documents_to_df(df_raw, history_classifier)
            df_with_docs["paragraph"] = df_with_docs.apply(
                lambda r: f"### 참고 문서 (Background Knowledge)\n{r['documents']}\n\n---\n### 문제 지문\n{r['paragraph']}"
                if pd.notna(r["documents"]) and r["documents"]
                else r["paragraph"],
                axis=1,
            )
            dataset.dataset["train"] = HFDataset.from_pandas(df_with_docs)
            print("학습 데이터 RAG 주입 완료")
        # =================================================================

        # Dataset 로드 및 전처리
        processed_dataset = dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **cfg.dataset.preprocess.train
        )

        # 📊 데이터셋을 8:2로 train/eval split
        from datasets import DatasetDict
        
        full_dataset = processed_dataset["train"]
        split_ratio = cfg.dataset.get("split_ratio", 0.8)  
        
        # train_test_split 사용
        split_dataset = full_dataset.train_test_split(
            train_size=split_ratio,
            seed=cfg.seed
        )
        
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        print("🚀 학습 모드 시작")
        print(f"📊 데이터셋 split 완료:")
        print(f"  - 학습 데이터: {len(train_dataset)}개 ({split_ratio*100:.0f}%)")
        print(f"  - 평가 데이터: {len(eval_dataset)}개 ({(1-split_ratio)*100:.0f}%)")
        
        # 학습 및 평가
        model.train(
            train_dataset=train_dataset,
            eval_dataset=None,
            **cfg.training
        )
        
    elif cfg.mode == "inference":
        print("🚀 추론 모드 시작")
        
        # 2-1. Model 초기화 (skip_init=True로 tokenizer만 로드)
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            skip_init=True,  # ⭐ 모델 초기화 스킵, 나중에 load_model()에서 로드
            max_seq_length=cfg.model.max_seq_length,
        )
        tokenizer = model.tokenizer
        
        # 2-2. 학습된 모델 로드
        model_load_path = cfg.inference.get("model_load_path", cfg.training.output_dir)
        if not os.path.exists(model_load_path):
            raise ValueError(f"모델 경로를 찾을 수 없습니다: {model_load_path}")
        # 체크포인트 디렉토리가 여러 개일 경우 가장 마지막 것을 로드
        p = Path(model_load_path)
        ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir()]
        if ckpts:
            ckpts.sort(
                key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
            )
            model_load_path = str(ckpts[-1])

        print(f"모델 로드 중: {model_load_path}")
        model.load_model(model_load_path)
        
        # 2-3. test.csv 로드 및 전처리
        test_dataset_path = cfg.inference.get("test_dataset_path", "data/test.csv")
        if not os.path.exists(test_dataset_path):
            raise ValueError(f"테스트 데이터셋 경로를 찾을 수 없습니다: {test_dataset_path}")
        
        print(f"테스트 데이터셋 로드 중: {test_dataset_path}")
        test_dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        test_dataset = test_dataset_cls(test_dataset_path)

        ### RAG부분 ###
        history_classifier = HistoryClassifier(model.model, tokenizer)
        # rag 설정이 있고 use가 True일 때만 retriever 주입
        if cfg.get("rag", {}).get("use", False):
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

            # 🔌 RAG 컨텍스트를 paragraph에 미리 주입
            # BaselineDataset은 self.dataset을 활용하므로, preprocess 전에 dataframe을 수정한다.
            from datasets import Dataset

            df_raw = test_dataset.dataset["train"].to_pandas()
            df_with_docs = rag_pipeline.add_documents_to_df(df_raw, history_classifier)
            df_with_docs["paragraph"] = df_with_docs.apply(
                lambda r: f"### 참고 문서 (Background Knowledge)\n{r['documents']}\n\n---\n### 문제 지문\n{r['paragraph']}"
                if pd.notna(r["documents"]) and r["documents"]
                else r["paragraph"],
                axis=1,
            )
            # Dataset으로 되돌려 세팅
            test_dataset.dataset = Dataset.from_pandas(df_with_docs)
        else:
            rag_pipeline = RAGPipeline(
                corpus_path="src/corpus/corpus.json",
                top_k=cfg.get("rag", {}).get("top_k", 5),
                retriever=None,
            )

        test_dataset.extra_columns["mode"] = cfg.mode
        test_dataset.extra_columns["history_classifier"] = history_classifier
        test_dataset.extra_columns["rag_pipeline"] = rag_pipeline

        processed_test_dataset = test_dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **cfg.dataset.preprocess.inference
        )
        
        # 2-4. 추론 수행
        predictions = model.predict(
            dataset=processed_test_dataset["train"],
            **cfg.inference
        )
        
        # 2-5. 결과 출력 (일부만)
        print(f"총 {len(predictions)}개 예측 완료")
        print(f"샘플 예측 결과: {list(predictions.items())[:3]}")
        
        # 2-6. output.csv 저장
        output_path = cfg.inference.get("output_path", "output/output.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # predictions 딕셔너리를 DataFrame으로 변환
        df_output = pd.DataFrame([
            {"id": id, "answer": answer} 
            for id, answer in predictions.items()
        ])
        df_output = df_output.sort_values("id")  # id 순서대로 정렬
        df_output.to_csv(output_path, index=False)
        print(f"결과 저장 완료: {output_path}")
        
    elif cfg.mode == "evaluate":
        print("🚀 평가 모드 시작")
        
        # Model 초기화 (skip_init=True로 tokenizer만 로드)
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            skip_init=True,  # ⭐ 모델 초기화 스킵, 나중에 load_model()에서 로드
            max_seq_length=cfg.model.max_seq_length,
        )

        tokenizer = model.tokenizer

        # 📊 train 모드와 동일하게 config의 path 데이터셋 로드
        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)
        
        # Dataset 로드 및 전처리
        processed_dataset = dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **cfg.dataset.preprocess.inference
        )
        
        # 📊 데이터셋을 8:2로 train/eval split (train과 동일한 방식)
        full_dataset = processed_dataset["train"]
        split_ratio = cfg.dataset.get("split_ratio", 0.8)
        
        # train_test_split 사용 (동일한 seed로 train과 같은 split)
        split_dataset = full_dataset.train_test_split(
            train_size=split_ratio,
            seed=cfg.seed
        )
        
        # 평가용 데이터만 사용 (20%)
        eval_dataset = split_dataset["test"]
        
        print(f"📊 평가 데이터셋 준비 완료:")
        print(f"  - 평가 데이터: {len(eval_dataset)}개 ({(1-split_ratio)*100:.0f}%)")

        # 학습된 모델 로드
        model_load_path = cfg.evaluate.get("model_load_path", cfg.training.output_dir)
        if not os.path.exists(model_load_path):
            raise ValueError(f"모델 경로를 찾을 수 없습니다: {model_load_path}")
        
        # 체크포인트 디렉토리가 여러 개일 경우 가장 마지막 것을 로드
        p = Path(model_load_path)
        ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir()]
        if ckpts:
            ckpts.sort(
                key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
            )
            model_load_path = str(ckpts[-1])
        
        print(f"모델 로드 중: {model_load_path}")
        model.load_model(model_load_path)
        
        # split된 평가 데이터로 평가
        metrics = model.evaluate(
            eval_dataset,
            original_dataset_path=cfg.dataset.path,  # 원본 데이터 경로 전달
            eval_output_path=cfg.evaluate.get("eval_output_path", "output/eval_results.csv")
        )
        print(f"평가 결과 : {metrics}")


if __name__ == "__main__":
    main()
