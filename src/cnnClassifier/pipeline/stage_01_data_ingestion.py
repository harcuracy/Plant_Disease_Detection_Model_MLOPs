from cnnClassifier import logger
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.config.configuration import ConfigurationManager


class DataIngestionTrainingPipeline:
    def main(self) -> None:
        logger.info(">>>>>> Stage 01: Data ingestion started <<<<<<")
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        dataset_path = data_ingestion.download_dataset()
        data_ingestion.create_subset(dataset_path)
        logger.info(">>>>>> Stage 01: Data ingestion completed <<<<<<")


if __name__ == "__main__":
    DataIngestionTrainingPipeline().main()
