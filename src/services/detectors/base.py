"""Giao diện chung cho detector 2D open-vocab."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection


class Detector(ABC):
    name: str = "base"

    def __init__(self, config: AutoLabelConfig):
        self.config = config
        self.prompt_to_class = config.prompt_to_class()
        self.prompts = list(self.prompt_to_class)

    @property
    def version(self) -> str:
        """Định danh model + tham số, dùng làm khoá cache."""
        return self.name

    @abstractmethod
    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        """Chạy trên một lô ảnh, trả về list detection cho từng ảnh."""

    def phrase_to_class(self, phrase: str) -> str | None:
        """Map cụm từ model trả về (có thể bị cắt/ghép) về lớp nội bộ."""
        phrase = phrase.strip().lower()
        if phrase in self.prompt_to_class:
            return self.prompt_to_class[phrase]
        for prompt, cls in self.prompt_to_class.items():
            if prompt in phrase or phrase in prompt:
                return cls
        words = set(phrase.split())
        best, best_overlap = None, 0
        for prompt, cls in self.prompt_to_class.items():
            overlap = len(words & set(prompt.split()))
            if overlap > best_overlap:
                best, best_overlap = cls, overlap
        return best


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
