"""GitHub公開前に、必須ファイルとNotebookの保存状態を検査する。"""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "notebooks/01_EDA.ipynb",
    "notebooks/02_Cancellation_Prediction.ipynb",
    "notebooks/03_SHAP_Analysis.ipynb",
    "notebooks/04_Business_Recommendations.ipynb",
    "notebooks/05_ROI_Sensitivity_Analysis.ipynb",
    "app.py",
    ".streamlit/config.toml",
    "src/roi_simulator.py",
    "tests/test_roi_simulator.py",
    "tests/test_streamlit_app.py",
    "sql/00_create_table.sql",
    "sql/01_load_data.sql",
    "sql/02_eda_queries.sql",
    "docker/docker-compose.yml",
    "docker/.env.example",
    "data/README.md",
    "images/lead_time_cancellation_rate.png",
    "images/market_segment_cancellation_rate.png",
    "images/shap_grouped_importance.png",
    "images/final_test_confusion_matrix.png",
    "images/roi_sensitivity.png",
    "images/roi_sensitivity_heatmap.png",
]


def main() -> None:
    problems: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (PROJECT_ROOT / relative_path).exists():
            problems.append(f"Missing: {relative_path}")

    for notebook_path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(notebook_path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        if "C:\\Users\\" in source:
            problems.append(f"Local absolute path: {notebook_path.name}")

        errors = [
            (index, output.get("ename"), output.get("evalue"))
            for index, cell in enumerate(notebook.cells)
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        if errors:
            problems.append(f"Saved execution errors: {notebook_path.name}: {errors}")

        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        nonempty_code_cells = [cell for cell in code_cells if cell.source.strip()]
        if any(cell.execution_count is None for cell in nonempty_code_cells):
            problems.append(f"Unexecuted code cell: {notebook_path.name}")

    if problems:
        print("Project validation failed:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("Project validation passed.")


if __name__ == "__main__":
    main()

