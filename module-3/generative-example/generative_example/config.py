from dataclasses import dataclass
from typing import Optional


@dataclass
class DataTrainingArguments:
    train_file: str
    test_file: str


@dataclass
class ModelArguments:
    model_id: str
    lora_r: int   
    lora_alpha: int
    lora_dropout: float
