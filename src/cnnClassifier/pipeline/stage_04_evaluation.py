from cnnClassifier import logger
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager


class EvaluationPipeline:
    def main(self) -> None:
        logger.info(">>>>>> Stage 04: Evaluation started <<<<<<")
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        logger.info(">>>>>> Stage 04: Evaluation completed <<<<<<")


if __name__ == "__main__":
    EvaluationPipeline().main()
