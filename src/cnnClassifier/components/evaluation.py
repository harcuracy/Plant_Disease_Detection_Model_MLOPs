import json

from cnnClassifier import logger
from cnnClassifier.entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.score = None

    def evaluation(self) -> None:
        tf = self._get_tensorflow()
        image_size = tuple(self.config.image_size[:2])
        logger.info(f"Loading trained model for evaluation: {self.config.trained_model_path}")

        model = tf.keras.models.load_model(self.config.trained_model_path)
        logger.info(f"Loading validation dataset from: {self.config.validation_data_dir}")
        validation_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.validation_data_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        validation_ds = validation_ds.map(lambda x, y: (preprocess_input(x), y))
        validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.score = model.evaluate(validation_ds)
        logger.info(f"Evaluation completed. Loss: {self.score[0]}, Accuracy: {self.score[1]}")
        self.save_score()

    def save_score(self) -> None:
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        with open(self.config.scores_path, "w", encoding="utf-8") as score_file:
            json.dump(scores, score_file, indent=4)
        logger.info(f"Evaluation scores saved: {self.config.scores_path}")

    @staticmethod
    def _get_tensorflow():
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tensorflow is required for evaluation. Install project dependencies with: uv sync"
            ) from exc
        return tf
