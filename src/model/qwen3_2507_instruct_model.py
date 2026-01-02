import re
import sys
from typing import Any, Dict, Optional
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
# Qwen3-2507 Thinking Model (Ollama 기반)
# ===========================================
@MODEL_REGISTRY.register("qwen3-ollama-instruct")
class Qwen3_2507InstructModel(BaseModelABC):
    """
    Ollama를 통해 Qwen3-2507 모델을 사용하는 추론 전용 모델.

    특징:
    - Ollama API 사용 (로컬 서빙)
    - think=True로 내부 추론 과정 활성화
    - Pydantic Structured Output으로 정답 강제
    """

    # super().__init__() 호출하지 않음
    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Args:
            model_name_or_path: Ollama 모델 이름 (예: "qwen3:30b-a3b")
        """

        logger.info("Qwen3_2507_InstructModel 초기화 중..")
        logger.debug(f"model_name_or_path: {model_name_or_path}")
        logger.debug(f"kwargs: {kwargs}")
        # BaseModel ABC는 model_name_or_path만 저장
        self.model_name = model_name_or_path  # Ollama는 별도 모델 객체 없음
        self.tokenizer = None  # Ollama는 토크나이저 불필요
        # Ollama 설정
        self.ollama_config = kwargs.get("ollama", {})
        logger.debug(f"self.model_name set to: {self.model_name}")
        logger.debug(f"ollama_config received: {self.ollama_config}")

        # TODO TEMPLATE Instruction 이용한 방법 공부할 것
        # 프롬프트 빌더 초기화
        logger.info("Initializing Qwen3ThinkingPromptBuilder...")
        self.prompt_builder = OllamaPromptBuilder()

        logger.info(f"Qwen3_2507_InstructModel initialized with model: {self.model_name}")

    # TODO ADAPTER 가능
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, **kwargs):
        """Ollama 모델은 학습을 지원하지 않음"""
        raise NotImplementedError("Qwen3_2507_ThinkingModel은 추론 전용입니다. 학습은 지원하지 않습니다.")

    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, float]:
        """Ollama 모델은 evaluate를 predict + metric 계산으로 대체"""
        raise NotImplementedError("evaluate는 predict 후 별도 metric 계산을 사용하세요.")

    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Ollama를 통한 추론 수행 (Structured Output 사용)

        Args:
            dataset: 추론할 데이터셋 (raw 형태 필요: paragraph, question, choices 등)
            **kwargs: 추론 설정
                - ollama_model: 사용할 Ollama 모델명
                - think: thinking 모드 활성화 여부 (기본값: True)
                - temperature: 생성 온도 (기본값: 0.6)
                - use_structured: Structured Output 사용 여부 (기본값: True)
                - template_name: 프롬프트 템플릿 이름 (기본값: qwen3_2507_thinking)

        Returns:
            Dict[str, Any]: {id: answer} 형태의 예측 결과
        """
        logger.info("Qwen3_2507_InstructModel: 시작 추론...")
        predictions = {}

        # 설정 로드
        # self.ollama_config = kwargs.get("ollama", {})
        # TODO: top_p, top_k 안쓰는데?
        # TODO formatting 을 쓸 것
        # Ollama 모델명은 ollama_config["model"]에서 가져옴
        model_name = self.ollama_config.get("model", self.model_name)
        temperature = self.ollama_config.get("temperature", 0.6)
        use_structured = self.ollama_config.get("use_structured", True)
        template_name = kwargs.get("template_name", "qwen3_2507_thinking")

        logger.debug(
            f"Predict settings - model_name: {model_name}, temperature: {temperature}, structured: {use_structured}"
        )

        # 프롬프트 빌더 설정
        if template_name != self.prompt_builder.template_name:
            self.prompt_builder = OllamaPromptBuilder(template_name=template_name)

        logger.info(f"Starting prediction with model: {model_name}")
        logger.info(f"Settings - temperature: {temperature}, structured: {use_structured}")

        # 데이터셋 순회 (tqdm과 logging 호환을 위해 logging_redirect_tqdm 사용)
        total = len(dataset) if hasattr(dataset, "__len__") else None

        # with logging_redirect_tqdm():
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
                    temperature=temperature,
                    use_structured=use_structured,
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
        temperature: float,
        use_structured: bool,
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
            logger.debug(f"[{row['id']}] System: {system_msg}")
            logger.debug(f"[{row['id']}] Prompt: {prompt_msg[:1000]}...")

        # Ollama 옵션
        options = {
            "temperature": temperature,
        }

        # TODO looger 반드시 설정해서 코드 개선하며 디버깅
        # TODO reasining 결과 저장하는 옵션
        # Ollama generate() 호출
        if use_structured:
            # Structured Output 사용 - JSON 스키마 강제
            response = generate(
                model=model_name,
                prompt=prompt_msg,
                system=system_msg,
                format=AnswerResponse.model_json_schema(),
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
                stream=False,
                options=options,
            )

            # 텍스트에서 숫자 추출
            answer = self._parse_answer_from_text(response.response)

        return answer

    def _parse_answer_from_text(self, text: str) -> int:
        """텍스트 응답에서 정답 숫자 추출"""
        text = text.strip()

        # 숫자만 있는 경우
        # TODO dict로 빠르게 처리하는 것도 고려
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
        logger.error("Qwen3_2507_InstructModel: save_model is not applicable for Ollama models")
        raise NotImplementedError("Ollama 모델은 저장 기능을 지원하지 않습니다.")

    def load_model(self, load_path: str):
        """Ollama 모델은 로드 불필요 (Ollama 서버에서 관리)"""
        logger.error("Qwen3_2507_InstructModel: load_model is not applicable for Ollama models")
        raise NotImplementedError("Ollama 모델은 로드 기능을 지원하지 않습니다.")
