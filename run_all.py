"""Run the complete exercise workflow from the repository root."""

from pathlib import Path
import subprocess
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parent


def run_script(relative_path: str) -> None:
    script = ROOT / relative_path
    print(f"Running {relative_path}...", flush=True)
    subprocess.run([sys.executable, script.name], cwd=script.parent, check=True)


def run_notebook(relative_path: str) -> None:
    notebook_path = ROOT / relative_path
    print(f"Running {relative_path}...", flush=True)
    with notebook_path.open(encoding="utf-8") as source:
        notebook = nbformat.read(source, as_version=4)

    executor = ExecutePreprocessor(timeout=900, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(notebook_path.parent)}})


def main() -> None:
    run_script("data/generate_synthetic_data.py")
    run_script("attribution/render_classification_flow.py")
    run_notebook("anomaly_detection/anomaly_detection.ipynb")
    run_notebook("attribution/attribution.ipynb")
    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
