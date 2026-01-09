from typing import Dict, Type, Any

class Registry:
    """
    모듈(Model, Dataset 등)을 등록하고 관리하는 레지스트리 클래스.
    YAML 설정 파일에서 문자열로 클래스를 호출할 수 있게 해줍니다.
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        """
        데코레이터로 클래스를 레지스트리에 등록합니다.
        
        사용 예시:
        @MODEL_REGISTRY.register("my_model")
        class MyModel(BaseModel):
            ...
        """
        def decorator(cls: Type):
            if name in self._registry:
                raise ValueError(f"'{name}'은(는) 이미 {self.name} 레지스트리에 등록되어 있습니다.")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type:
        """등록된 클래스를 이름으로 가져옵니다."""
        if name not in self._registry:
            raise KeyError(
                f"'{name}'은(는) {self.name} 레지스트리에 없습니다. "
                f"사용 가능한 옵션: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_available(self):
        """등록된 모든 클래스 이름을 반환합니다."""
        return list(self._registry.keys())

# 전역 레지스트리 객체 생성
MODEL_REGISTRY = Registry("Model")
DATASET_REGISTRY = Registry("Dataset")
METRIC_REGISTRY = Registry("Metric")
