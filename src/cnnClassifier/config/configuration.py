from pathlib import Path

from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from cnnClassifier import logger
from cnnClassifier.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])
        logger.info("Configuration manager initialized")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.data_ingestion

        create_directories([Path(config.root_dir), Path(config.raw_data_dir), Path(config.subset_data_dir)])
        logger.info("Data ingestion configuration created")

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_URL,
            raw_data_dir=Path(config.raw_data_dir),
            subset_data_dir=Path(config.subset_data_dir),
            validation_split=params.validation_split,
            max_images_per_class=params.max_images_per_class,
            random_seed=params.random_seed,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.prepare_base_model

        create_directories([Path(config.root_dir)])
        logger.info("Prepare base model configuration created")

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            training_data_dir=Path(config.training_data_dir),
            image_size=list(params.image_size),
            learning_rate=params.learning_rate,
            include_top=params.include_top,
            weights=params.weights,
            classes=self._resolve_class_count(Path(config.training_data_dir), params.classes),
            dropout_rate=params.dropout_rate,
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        create_directories([Path(config.root_dir)])
        logger.info("Training configuration created")

        return TrainingConfig(
            root_dir=Path(config.root_dir),
            training_data_dir=Path(config.training_data_dir),
            validation_data_dir=Path(config.validation_data_dir),
            trained_model_path=Path(config.trained_model_path),
            class_names_path=Path(config.class_names_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            image_size=list(params.prepare_base_model.image_size),
            batch_size=params.training.batch_size,
            epochs=params.training.epochs,
            augmentation=params.training.augmentation,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        params = self.params

        create_directories([Path(config.root_dir)])
        logger.info("Evaluation configuration created")

        return EvaluationConfig(
            root_dir=Path(config.root_dir),
            validation_data_dir=Path(config.validation_data_dir),
            trained_model_path=Path(config.trained_model_path),
            scores_path=Path(config.scores_path),
            image_size=list(params.prepare_base_model.image_size),
            batch_size=params.training.batch_size,
        )

    @staticmethod
    def _resolve_class_count(training_data_dir: Path, default_classes: int) -> int:
        if not training_data_dir.exists():
            logger.info(f"Training data directory not found. Using default class count: {default_classes}")
            return default_classes

        class_dirs = [path for path in training_data_dir.iterdir() if path.is_dir()]
        resolved_classes = len(class_dirs) or default_classes
        logger.info(f"Resolved class count: {resolved_classes}")
        return resolved_classes
