from dataclasses import dataclass
from typing import Optional


@dataclass
class DataTrainingArguments:
    train_file: str
    validation_file: str
    max_seq_length: int = 128
    overwrite_cache: bool = False
    pad_to_max_length: bool = True
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


@dataclass
class ModelArguments:
    model_name_or_path: str
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    use_wandb: bool = False
    save_model: bool = False
