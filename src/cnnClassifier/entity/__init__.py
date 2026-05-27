from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    raw_data_dir: Path
    subset_data_dir: Path
    validation_split: float
    max_images_per_class: int
    random_seed: int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    training_data_dir: Path
    image_size: list[int]
    learning_rate: float
    include_top: bool
    weights: str
    classes: int
    dropout_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    training_data_dir: Path
    validation_data_dir: Path
    trained_model_path: Path
    class_names_path: Path
    updated_base_model_path: Path
    image_size: list[int]
    batch_size: int
    epochs: int
    augmentation: bool


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    validation_data_dir: Path
    trained_model_path: Path
    scores_path: Path
    image_size: list[int]
    batch_size: int
