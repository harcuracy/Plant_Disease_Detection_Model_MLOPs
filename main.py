from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline


if __name__ == "__main__":
    logger.info("Pipeline started")
    DataIngestionTrainingPipeline().main()
    PrepareBaseModelTrainingPipeline().main()
    ModelTrainingPipeline().main()
    EvaluationPipeline().main()
    logger.info("Pipeline completed")
