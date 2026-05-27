from pathlib import Path

import numpy as np
from PIL import Image

from cnnClassifier import logger
from cnnClassifier.utils.common import read_yaml


class PredictionService:
    def __init__(
        self,
        model_path: Path,
        class_names_path: Path,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.image_size = image_size
        self.model = None
        self.class_names: list[str] = []

    def load_assets(self) -> None:
        if self.model is not None and self.class_names:
            return

        tf = self._get_tensorflow()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}. Run the training pipeline first.")
        if not self.class_names_path.exists():
            raise FileNotFoundError(
                f"Class names not found: {self.class_names_path}. Run the training pipeline first."
            )

        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = list(read_yaml(self.class_names_path).class_names)
        logger.info("Prediction assets loaded")

    def decode_image(self, image_file) -> Image.Image:
        image = Image.open(image_file).convert("RGB")
        return image.resize(self.image_size)

    def predict(self, image: Image.Image) -> dict:
        tf = self._get_tensorflow()
        self.load_assets()

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

        predictions = self.model.predict(image_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            "class": self.class_names[predicted_index],
            "confidence": round(confidence, 4),
        }

    @staticmethod
    def _get_tensorflow():
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tensorflow is required for prediction. Install dependencies with: pip install -r requirements.txt"
            ) from exc
        return tf
