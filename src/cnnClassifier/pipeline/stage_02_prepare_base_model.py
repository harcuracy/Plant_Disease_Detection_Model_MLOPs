from cnnClassifier import logger
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager


class PrepareBaseModelTrainingPipeline:
    def main(self) -> None:
        logger.info(">>>>>> Stage 02: Prepare base model started <<<<<<")
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        logger.info(">>>>>> Stage 02: Prepare base model completed <<<<<<")


if __name__ == "__main__":
    PrepareBaseModelTrainingPipeline().main()
