import logging
import os
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(message)s]",
)

project_name = "cnnClassifier"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "pyproject.toml",
    "research/trials.ipynb",
    "templates/index.html",
    "tests/__init__.py",
    ".github/workflows/ecr.yml",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    "Dockerfile",
    "README.md",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info("Creating directory: %s for the file: %s", filedir, filename)

    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info("Creating empty file: %s", filepath)
    else:
        logging.info("%s already exists", filename)
