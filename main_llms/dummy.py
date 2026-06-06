from typing import Any, Dict


class DummyLLM:
    """Offline debug LLM. It avoids API calls and always returns an incorrect answer."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def generate(self, prompt: str) -> str:
        return "I cannot solve this in dummy mode.\n#### 0"
