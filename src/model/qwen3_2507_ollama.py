"""
Qwen3-2507 Ollama 통합 모델 모듈.

=== Ollama Structured Output 동작 방식 ===
[조사 일자: 2025.01]
[참고: https://blog.danielclayton.co.uk/posts/ollama-structured-outputs/]

1. 내부 구현
   - llama.cpp의 GBNF (Generalized BNF) grammar 사용
   - JSON schema를 GBNF grammar로 자동 변환 (Ollama v0.5+)

2. Logit Masking 방식
   - 모델이 토큰 생성 시, grammar에 맞지 않는 토큰의 logit을 -INFINITY로 설정
   - 샘플링 단계에서 해당 토큰들이 선택되지 않음
   - 예: '{'를 생성 후 → '"' 또는 '}' 만 선택 가능

3. 특징
   - 모델은 masking 존재를 인식하지 못함
   - grammar 샘플링은 추론 속도 저하 가능
   - 프롬프트에 JSON 형식 명시 권장 (모델 문맥 파악 용이)

4. 주의사항
   - 응답 중간에 끊기면 불완전한 JSON 발생 가능
   - Ollama는 전체 응답의 schema 유효성 검증 안 함

=== num_predict와 thinking 모드 ===
[불명확한 사항]
- thinking 토큰이 num_predict에 포함되는지 공식 문서에 미명시
- eval_count는 "응답의 토큰 수"로만 정의

[권장]
- thinking 모드: num_predict를 충분히 크게 (-1 권장)
- instruct + structured output: 작은 값도 가능

=== Qwen3-2507 Best Practices ===
[참고: https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-2507]

| 파라미터    | Instruct | Thinking |
|------------|----------|----------|
| temperature| 0.7      | 0.6      |
| top_p      | 0.8      | 0.95     |
| top_k      | 20       | 20       |
| min_p      | 0.0      | 0.0      |
"""

import re
import sys
from typing import Any, Dict, Optional, Set
from datasets import Dataset
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Literal

from ollama import generate

from src.model.base_model import BaseModel as BaseModelABC
from src.utils.registry import MODEL_REGISTRY
from src.utils import get_logger
from src.prompt.ollama_prompt import OllamaPromptBuilder

logger = get_logger(__name__)


# ===========================================
# 유효한 Ollama Options 목록
# https://github.com/ollama/ollama-python/blob/main/ollama/_types.py
# ===========================================
VALID_OPTIONS: Set[str] = {
    # 컨텍스트/생성 제어
    "num_ctx",  # 컨텍스트 윈도우 크기 (기본 2048)
    "num_predict",  # 최대 생성 토큰 수 (-1: 무제한, -2: 컨텍스트 채우기)
    "num_batch",  # 배치 크기
    # 샘플링
    "temperature",  # 창의성 (기본 0.8, 높을수록 창의적)
    "top_p",  # 누적 확률 기반 샘플링 (기본 0.9)
    "top_k",  # 상위 k개 토큰만 고려 (기본 40)
    "min_p",  # 최소 확률 임계값 (기본 0.0)
    # 반복 제어
    "repeat_penalty",  # 반복 페널티 (기본 1.1)
    "repeat_last_n",  # 반복 체크 범위 (기본 64)
    "presence_penalty",  # 존재 페널티 (0.0~2.0)
    "frequency_penalty",  # 빈도 페널티
    # 재현성
    "seed",  # 재현성을 위한 시드 (0: 랜덤)
    "stop",  # 생성 중단 시퀀스
    # 고급 (Mirostat)
    "mirostat",  # Mirostat 알고리즘 (0: off, 1: v1, 2: v2)
    "mirostat_tau",  # Mirostat 타우값
    "mirostat_eta",  # Mirostat 에타값
}


# ===========================================
# Pydantic Response Schema for Structured Output
# ===========================================
class AnswerResponse(BaseModel):
    """객관식 정답 응답 스키마 (Structured Output용)"""

    answer: Literal[1, 2, 3, 4, 5] = Field(description="선택한 정답 번호 (1-5)")


class AnswerWithReasoning(BaseModel):
    """추론 과정을 포함한 응답 스키마"""

    reasoning: str = Field(description="정답을 선택한 이유")
    answer: Literal[1, 2, 3, 4, 5] = Field(description="선택한 정답 번호 (1-5)")


# ===========================================
# Qwen3-2507 통합 Ollama 모델
# ===========================================
@MODEL_REGISTRY.register("Qwen3-2507_ollama")
class Qwen3_2507OllamaModel(BaseModelABC):
    """
    통합 Qwen3-2507 Ollama 모델.

    Config 구조:
    - model.variant: 'thinking' | 'instruct' (모델 종류)
    - model.name: Ollama 모델명
    - ollama.think: bool (thinking 기능 활성화)
    - ollama.use_structured: bool (Pydantic 스키마로 응답 강제)
    - ollama.options: dict (생성 파라미터)

    사용 예:
    - thinking 모델 + think=True: 추론 과정 포함
    - thinking 모델 + think=False: 추론 과정 없이 빠른 답변
    - instruct 모델: 일반적으로 think=False
    """

    # 필수 옵션 목록 (config에 반드시 있어야 함)
    # 이 옵션들이 config에 명시되지 않으면 프로그램 종료
    REQUIRED_OPTIONS = {
        "num_ctx",
        "num_predict",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "seed",
    }

    def __init__(self, cfg):
        """
        Args:
            cfg: Hydra DictConfig 객체 (전체 설정)
                - cfg.model: 모델 관련 설정 (type, variant, model_name_or_path)
                - cfg.ollama: Ollama 관련 설정 (think, use_structured, options)
                - cfg.prompt: 프롬프트 관련 설정 (name)
        """
        from omegaconf import OmegaConf

        logger.info("Qwen3_2507OllamaModel 초기화 중...")

        # cfg 저장 (필요시 다른 설정 접근용)
        self.cfg = cfg
        self.tokenizer = None  # Ollama는 토크나이저 불필요

        # ========================================
        # 필수 설정 검증 (없으면 즉시 에러)
        # ========================================

        # model 설정 검증
        if "model" not in cfg:
            raise ValueError("cfg.model 설정이 없습니다.")

        model_cfg = cfg.model
        if "model_name_or_path" not in model_cfg:
            raise ValueError("cfg.model.model_name_or_path 설정이 없습니다.")
        if "variant" not in model_cfg:
            raise ValueError("cfg.model.variant 설정이 없습니다. ('thinking' 또는 'instruct')")

        self.model_name = model_cfg.model_name_or_path
        self.variant = model_cfg.variant

        if self.variant not in ("thinking", "instruct"):
            raise ValueError(f"cfg.model.variant는 'thinking' 또는 'instruct'여야 합니다. 현재값: {self.variant}")

        logger.debug(f"model_name: {self.model_name}")
        logger.debug(f"variant: {self.variant}")

        # ollama 설정 검증
        if "ollama" not in cfg:
            raise ValueError("cfg.ollama 설정이 없습니다.")

        self.ollama_config = OmegaConf.to_container(cfg.ollama, resolve=True)
        logger.debug(f"ollama_config: {self.ollama_config}")

        # prompt 설정 검증
        if "prompt" not in cfg:
            raise ValueError("cfg.prompt 설정이 없습니다.")
        if "name" not in cfg.prompt:
            raise ValueError("cfg.prompt.name 설정이 없습니다.")

        self.prompt_config = OmegaConf.to_container(cfg.prompt, resolve=True)
        template_name = self.prompt_config["name"]

        logger.info(f"prompt_config: {self.prompt_config}")
        logger.info(f"Using template: {template_name}")
        self.prompt_builder = OllamaPromptBuilder(template_name=template_name)

        logger.info(f"Qwen3_2507OllamaModel initialized - variant: {self.variant}, model: {self.model_name}")

    def _build_ollama_options(self) -> dict:
        """
        config의 ollama.options를 ollama API options로 변환.

        필수 옵션이 없으면 즉시 에러 발생.

        Returns:
            dict: ollama generate()에 전달할 options 딕셔너리
        """
        # options 설정 검증
        if "options" not in self.ollama_config:
            raise ValueError("ollama.options 설정이 없습니다.")

        config_options = self.ollama_config["options"]

        # 필수 옵션 검증
        missing_options = self.REQUIRED_OPTIONS - set(config_options.keys())
        if missing_options:
            raise ValueError(f"ollama.options에 필수 옵션이 없습니다: {missing_options}")

        # 유효한 옵션만 추출
        options = {}
        for key, value in config_options.items():
            if key in VALID_OPTIONS:
                if value is None:
                    raise ValueError(f"ollama.options.{key} 값이 None입니다.")
                options[key] = value
            else:
                raise ValueError(f"올바르지 않은 ollama option: {key}")

        logger.debug(f"Built ollama options: {options}")
        return options

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, **kwargs):
        """Ollama 모델은 학습을 지원하지 않음"""
        raise NotImplementedError("Qwen3_2507OllamaModel은 추론 전용입니다. 학습은 지원하지 않습니다.")

    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, float]:
        """Ollama 모델은 evaluate를 predict + metric 계산으로 대체"""
        raise NotImplementedError("evaluate는 predict 후 별도 metric 계산을 사용하세요.")

    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Ollama를 통한 추론 수행 (Structured Output 사용)

        Args:
            dataset: 추론할 데이터셋 (raw 형태 필요: paragraph, question, choices 등)
            **kwargs: 추론 설정

        Returns:
            Dict[str, Any]: {id: answer} 형태의 예측 결과
        """
        logger.info(f"Qwen3_2507OllamaModel: 추론 시작 (variant: {self.variant})")
        predictions = {}

        # 설정 검증 및 로드
        # model: ollama.model이 있으면 사용, 없으면 model.model_name_or_path 사용
        model_name = self.ollama_config.get("model", self.model_name)

        if "think" not in self.ollama_config:
            raise ValueError("ollama.think 설정이 없습니다.")
        if "use_structured" not in self.ollama_config:
            raise ValueError("ollama.use_structured 설정이 없습니다.")

        use_think = self.ollama_config["think"]
        use_structured = self.ollama_config["use_structured"]

        # options 빌드
        options = self._build_ollama_options()

        logger.info(f"Model: {model_name}")
        logger.info(f"Settings - variant: {self.variant}, think: {use_think}, structured: {use_structured}")
        logger.info(f"Options: {options}")

        # 데이터셋 순회
        total = len(dataset) if hasattr(dataset, "__len__") else None

        for idx, row in enumerate(
            tqdm(
                dataset,
                desc="Predicting",
                total=total,
                dynamic_ncols=True,
                unit="문제",
                mininterval=10.0,
                disable=not sys.stdout.isatty(),
            ),
            start=1,
        ):
            try:
                answer = self._predict_single(
                    row=row,
                    model_name=model_name,
                    use_think=use_think,
                    use_structured=use_structured,
                    options=options,
                    counter=idx,
                )
                predictions[row["id"]] = answer

            except Exception as e:
                logger.warning(f"Error predicting id={row['id']}: {e}")
                raise e

        return predictions

    def _predict_single(
        self,
        row: Dict[str, Any],
        model_name: str,
        use_think: bool,
        use_structured: bool,
        options: dict,
        counter: int,
    ) -> int:
        """단일 문제에 대한 예측 수행"""

        # 프롬프트 빌드 (messages 형식으로 반환됨)
        messages = self.prompt_builder.build_single(
            paragraph=row["paragraph"],
            question=row["question"],
            choices=row["choices"],
            question_plus=row.get("question_plus"),
        )

        # messages에서 system과 prompt 분리
        system_msg = ""
        prompt_msg = ""
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                prompt_msg = msg["content"]

        if counter % 10 == 0:
            logger.debug(f"[{row['id']}] System: {system_msg[:200]}...")
            logger.debug(f"[{row['id']}] Prompt: {prompt_msg[:500]}...")

        # TODO response 항목 반드시 로그로 출력해보기
        # Ollama generate() 호출
        if use_structured:
            # Structured Output 사용 - JSON 스키마 강제
            # [참고] GBNF grammar + logit masking으로 {"answer": 1~5} 형태만 출력
            response = generate(
                model=model_name,
                prompt=prompt_msg,
                system=system_msg,
                format=AnswerResponse.model_json_schema(),
                think=use_think,
                stream=False,
                options=options,
            )

            # JSON 파싱 (generate는 response.response 사용)
            result = AnswerResponse.model_validate_json(response.response)
            answer = result.answer

        else:
            # 일반 텍스트 응답
            response = generate(
                model=model_name,
                prompt=prompt_msg,
                system=system_msg,
                think=use_think,
                stream=False,
                options=options,
            )

            # 텍스트에서 숫자 추출
            answer = self._parse_answer_from_text(response.response)

        # 디버그 로깅 (thinking 내용)
        if use_think and hasattr(response, "thinking") and response.thinking:
            logger.debug(f"[{row['id']}] Thinking: {response.thinking[:200]}...")

        return answer

    def _parse_answer_from_text(self, text: str) -> int:
        """텍스트 응답에서 정답 숫자 추출"""
        text = text.strip()

        # 숫자만 있는 경우
        logger.debug(f"text received for parsing: {text}")
        if text in ["1", "2", "3", "4", "5"]:
            logger.debug(f"Parsed answer directly: {text}")
            return int(text)

        # 정규식으로 숫자 추출
        match = re.search(r"[1-5]", text)
        if match:
            logger.debug(f"Parsed answer from regex: {match.group()}")
            return int(match.group())

        # 파싱 실패 시 기본값
        logger.warning(f"Failed to parse answer from: {text[:100]}")
        return 1

    def save_model(self, save_path: str):
        """Ollama 모델은 저장 불필요"""
        logger.error("Qwen3_2507OllamaModel: save_model is not applicable for Ollama models")
        raise NotImplementedError("Ollama 모델은 저장 기능을 지원하지 않습니다.")

    def load_model(self, load_path: str):
        """Ollama 모델은 로드 불필요 (Ollama 서버에서 관리)"""
        logger.error("Qwen3_2507OllamaModel: load_model is not applicable for Ollama models")
        raise NotImplementedError("Ollama 모델은 로드 기능을 지원하지 않습니다.")
