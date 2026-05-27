from cnnClassifier import logger
from cnnClassifier.entity import TrainingConfig
from cnnClassifier.utils.common import save_yaml


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_ds = None
        self.validation_ds = None

    def get_base_model(self) -> None:
        tf = self._get_tensorflow()
        logger.info(f"Loading updated base model: {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        logger.info("Updated base model loaded")

    def train_valid_generator(self) -> None:
        tf = self._get_tensorflow()
        image_size = tuple(self.config.image_size[:2])
        logger.info(f"Loading training dataset from: {self.config.training_data_dir}")

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.training_data_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        logger.info(f"Loading validation dataset from: {self.config.validation_data_dir}")

        self.validation_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.validation_data_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        save_yaml(self.config.class_names_path, {"class_names": self.train_ds.class_names})
        logger.info(f"Class names saved: {self.config.class_names_path}")
        self.train_ds = self._prepare_dataset(self.train_ds, training=True)
        self.validation_ds = self._prepare_dataset(self.validation_ds, training=False)
        logger.info("Training and validation datasets prepared")

    def train(self) -> None:
        if self.model is None:
            self.get_base_model()
        if self.train_ds is None or self.validation_ds is None:
            self.train_valid_generator()

        self.model.fit(
            self.train_ds,
            epochs=self.config.epochs,
            validation_data=self.validation_ds,
        )
        self.save_model(path=self.config.trained_model_path, model=self.model)
        logger.info(f"Trained model saved: {self.config.trained_model_path}")

    def _prepare_dataset(self, dataset, training: bool):
        tf = self._get_tensorflow()
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        if training and self.config.augmentation:
            augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomZoom(0.1),
                ]
            )

            dataset = dataset.map(lambda x, y: (preprocess_input(augmentation(x, training=True)), y))
        else:
            dataset = dataset.map(lambda x, y: (preprocess_input(x), y))

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    @staticmethod
    def save_model(path, model) -> None:
        model.save(path)

    @staticmethod
    def _get_tensorflow():
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tensorflow is required for training. Install project dependencies with: uv sync"
            ) from exc
        return tf
