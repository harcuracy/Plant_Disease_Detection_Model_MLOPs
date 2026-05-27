import random
import shutil
from pathlib import Path

from cnnClassifier import logger
from cnnClassifier.entity import DataIngestionConfig


class DataIngestion:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    TRAIN_DIR_NAMES = {"train", "training"}
    NON_CLASS_DIR_NAMES = {"test", "valid", "validation"}

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self) -> Path:
        logger.info(f"Downloading Kaggle dataset: {self.config.source_url}")
        try:
            import kagglehub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "kagglehub is required to download the dataset. Install it with: pip install kagglehub"
            ) from exc

        downloaded_path = Path(kagglehub.dataset_download(self.config.source_url))
        raw_dataset_path = self.config.raw_data_dir / downloaded_path.name

        if raw_dataset_path.exists():
            logger.info(f"Raw dataset already exists: {raw_dataset_path}")
            return raw_dataset_path

        shutil.copytree(downloaded_path, raw_dataset_path)
        logger.info(f"Raw dataset copied to: {raw_dataset_path}")
        return raw_dataset_path

    def create_subset(self, dataset_path: Path) -> None:
        logger.info(f"Creating subset from dataset path: {dataset_path}")
        class_dirs = self._find_class_dirs(dataset_path)
        logger.info(f"Found {len(class_dirs)} class directories")
        self._reset_subset_dir()

        for class_dir in class_dirs:
            images = self._list_images(class_dir)
            random.Random(self.config.random_seed).shuffle(images)

            selected_images = images[: self.config.max_images_per_class]
            split_index = int(len(selected_images) * (1 - self.config.validation_split))

            self._copy_images(selected_images[:split_index], "train", class_dir.name)
            self._copy_images(selected_images[split_index:], "validation", class_dir.name)
            logger.info(
                f"Subset created for {class_dir.name}: "
                f"{split_index} train, {len(selected_images) - split_index} validation"
            )

        logger.info(f"Subset dataset saved to: {self.config.subset_data_dir}")

    def _find_class_dirs(self, dataset_path: Path) -> list[Path]:
        training_dirs = [
            path
            for path in dataset_path.rglob("*")
            if path.is_dir() and path.name.lower() in self.TRAIN_DIR_NAMES and self._has_class_children(path)
        ]

        if training_dirs:
            training_dir = sorted(training_dirs, key=lambda path: len(path.parts))[0]
            logger.info(f"Using training directory for class discovery: {training_dir}")
            return sorted(path for path in training_dir.iterdir() if path.is_dir() and self._list_images(path))

        class_dirs = []
        for path in dataset_path.rglob("*"):
            if path.is_dir() and path.name.lower() not in self.NON_CLASS_DIR_NAMES and self._list_images(path):
                class_dirs.append(path)

        leaf_class_dirs = [
            path for path in class_dirs if not any(child in class_dirs for child in path.iterdir() if child.is_dir())
        ]
        return sorted(leaf_class_dirs)

    def _has_class_children(self, path: Path) -> bool:
        return any(child.is_dir() and self._list_images(child) for child in path.iterdir())

    def _list_images(self, path: Path) -> list[Path]:
        return sorted(
            file_path
            for file_path in path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def _reset_subset_dir(self) -> None:
        if self.config.subset_data_dir.exists():
            shutil.rmtree(self.config.subset_data_dir)
            logger.info(f"Removed existing subset directory: {self.config.subset_data_dir}")
        self.config.subset_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created subset directory: {self.config.subset_data_dir}")

    def _copy_images(self, image_paths: list[Path], split: str, class_name: str) -> None:
        target_dir = self.config.subset_data_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            shutil.copy2(image_path, target_dir / image_path.name)
