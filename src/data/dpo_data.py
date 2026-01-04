import pandas as pd
from datasets import Dataset
from typing import Optional, Dict, Any
from src.data.base_data import BaseDataset
from src.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("dpo")
class DPODataset(BaseDataset):
    """
    DPO(Direct Preference Optimization) 학습을 위한 데이터셋 클래스.
    
    CSV 형식:
    - id: 샘플 ID
    - prompt: 지문 + 문제 + 선택지 (전체 프롬프트)
    - chosen: 선호되는 답변 (정답 설명)
    - rejected: 선호되지 않는 답변 (오답 설명)
    """
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.data = self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """
        CSV 파일에서 DPO 데이터 로드
        BaseDataset의 추상 메서드 구현
        """
        df = pd.read_csv(self.data_path)
        
        # 필수 컬럼 확인
        required_cols = ["prompt", "chosen", "rejected"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DPO 데이터셋에 필수 컬럼이 없습니다: {missing_cols}")
        
        # 결측치 제거
        original_len = len(df)
        df = df.dropna(subset=required_cols)
        
        if len(df) < original_len:
            print(f"⚠️  결측치 제거: {original_len - len(df)}개 샘플 제외")
        
        print(f"✅ DPO 데이터셋 로드 완료: {len(df)}개 샘플")
        
        # 데이터 샘플 출력 (디버깅용)
        if len(df) > 0:
            print(f"\n📝 데이터 샘플:")
            print(f"   - ID: {df['id'].iloc[0] if 'id' in df.columns else 'N/A'}")
            print(f"   - Prompt 길이: {len(df['prompt'].iloc[0])} 문자")
            print(f"   - Prompt 시작: {df['prompt'].iloc[0][:100]}...")
            print(f"   - Chosen 길이: {len(df['chosen'].iloc[0])} 문자")
            print(f"   - Chosen 시작: {df['chosen'].iloc[0][:80]}...")
            print(f"   - Rejected 길이: {len(df['rejected'].iloc[0])} 문자")
            print(f"   - Rejected 시작: {df['rejected'].iloc[0][:80]}...")
        
        return df
    
    def preprocess(
        self,
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: Optional[int] = None,
        template: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Dataset]:
        """
        DPO 학습을 위한 데이터 전처리
        
        실제 데이터 형식:
        - prompt: "### 지문:\n...\n\n### 문제:\n...\n\n### 선택지:\n..."
        - chosen: "정답에 대한 설명 (선호되는 답변)"
        - rejected: "오답에 대한 설명 (선호되지 않는 답변)"
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: 전체 시퀀스 최대 길이
            max_prompt_length: prompt 부분만의 최대 길이 (None이면 max_length의 절반)
            template: 프롬프트 템플릿 이름 (사용하지 않음)
            **kwargs: 추가 설정
        
        Returns:
            {"train": Dataset} 형태의 딕셔너리
        """
        if max_prompt_length is None:
            max_prompt_length = max_length // 2
        
        print(f"\n🔧 DPO 데이터 전처리 시작...")
        print(f"   - max_length: {max_length}")
        print(f"   - max_prompt_length: {max_prompt_length}")
        
        # DPO 데이터 포맷 생성
        dpo_data = []
        skipped = 0
        
        for idx, row in self.data.iterrows():
            prompt = str(row["prompt"]).strip()
            chosen = str(row["chosen"]).strip()
            rejected = str(row["rejected"]).strip()
            
            # 빈 값 체크
            if not prompt or not chosen or not rejected:
                skipped += 1
                continue
            
            # Unsloth DPO 포맷: prompt, chosen, rejected 필드
            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        
        if skipped > 0:
            print(f"   ⚠️  빈 값으로 인해 {skipped}개 샘플 제외")
        
        # Dataset 객체로 변환
        dataset = Dataset.from_list(dpo_data)
        
        print(f"✅ DPO 데이터셋 전처리 완료: {len(dataset)}개")
        
        # 샘플 데이터 출력 (디버깅)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n📊 전처리된 샘플 예시:")
            print(f"   - Prompt: {sample['prompt'][:150]}...")
            print(f"   - Chosen: {sample['chosen'][:100]}...")
            print(f"   - Rejected: {sample['rejected'][:100]}...")
        
        return {"train": dataset}
    
    def get_raw_data(self) -> pd.DataFrame:
        """원본 데이터프레임 반환"""
        return self.data
