from cnnClassifier import logger
from cnnClassifier.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None

    def get_base_model(self) -> None:
        tf = self._get_tensorflow()
        logger.info("Preparing MobileNetV2 base model")

        self.model = tf.keras.applications.MobileNetV2(
            input_shape=tuple(self.config.image_size),
            weights=self.config.weights,
            include_top=self.config.include_top,
        )
        self.save_model(path=self.config.base_model_path, model=self.model)
        logger.info(f"Base model saved: {self.config.base_model_path}")

    def update_base_model(self) -> None:
        tf = self._get_tensorflow()
        logger.info(f"Updating base model for {self.config.classes} classes")

        if self.model is None:
            self.model = tf.keras.models.load_model(self.config.base_model_path)
            logger.info(f"Base model loaded: {self.config.base_model_path}")

        self.model.trainable = False

        model = tf.keras.Sequential(
            [
                self.model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.save_model(path=self.config.updated_base_model_path, model=model)
        logger.info(f"Updated base model saved: {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(path, model) -> None:
        model.save(path)

    @staticmethod
    def _get_tensorflow():
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tensorflow is required for model preparation. Install project dependencies with: pip install -r requirements.txt"
            ) from exc
        return tf
