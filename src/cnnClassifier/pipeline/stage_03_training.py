from cnnClassifier import logger
from cnnClassifier.components.training import Training
from cnnClassifier.config.configuration import ConfigurationManager


class ModelTrainingPipeline:
    def main(self) -> None:
        logger.info(">>>>>> Stage 03: Training started <<<<<<")
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()
        logger.info(">>>>>> Stage 03: Training completed <<<<<<")


if __name__ == "__main__":
    ModelTrainingPipeline().main()
