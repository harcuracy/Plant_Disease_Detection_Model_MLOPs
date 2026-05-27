import json
from pathlib import Path


def test_research_notebook_is_valid_json():
    notebook_path = Path("research/trials.ipynb")

    with notebook_path.open(encoding="utf-8") as file:
        notebook = json.load(file)

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
