"""実行済みNotebookのPNG出力をREADME用のimages/へ書き出す。"""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
import nbformat
import pandas as pd
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
IMAGE_DIR = PROJECT_ROOT / "images"

FIGURE_SPECS = [
    {
        "notebook": "04_Business_Recommendations.ipynb",
        "source_text": "fig, ax1 = plt.subplots(figsize=(11, 5))",
        "image_index": 0,
        "filename": "lead_time_cancellation_rate.png",
    },
    {
        "notebook": "03_SHAP_Analysis.ipynb",
        "source_text": "def build_original_feature_labels",
        "image_index": 1,
        "filename": "shap_grouped_importance.png",
    },
    {
        "notebook": "02_Cancellation_Prediction.ipynb",
        "source_text": "final_confusion_matrix = confusion_matrix",
        "image_index": 0,
        "filename": "final_test_confusion_matrix.png",
    },
    {
        "notebook": "04_Business_Recommendations.ipynb",
        "source_text": "reduction_rate_grid = np.linspace",
        "image_index": 0,
        "filename": "roi_sensitivity.png",
    },
]


def create_market_segment_figure() -> None:
    """主要セグメントのキャンセル率を、実データからREADME向けに描画する。"""
    data_path = PROJECT_ROOT / "data" / "hotel_bookings.csv"
    bookings = pd.read_csv(data_path)

    segment_summary = (
        bookings.groupby("market_segment", dropna=False)["is_canceled"]
        .agg(total_bookings="size", cancellation_rate="mean")
        .nlargest(6, "total_bookings")
        .sort_values("cancellation_rate")
    )

    colors = [
        "#E07A5F" if segment == "Groups" else "#4C78A8"
        for segment in segment_summary.index
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(segment_summary.index, segment_summary["cancellation_rate"], color=colors)
    ax.axvline(
        bookings["is_canceled"].mean(),
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label="Overall cancellation rate",
    )
    ax.bar_label(
        bars,
        labels=[
            f"{rate:.1%}  (n={count:,})"
            for rate, count in zip(
                segment_summary["cancellation_rate"],
                segment_summary["total_bookings"],
            )
        ],
        padding=4,
        fontsize=9,
    )
    ax.set_xlim(0, max(0.75, segment_summary["cancellation_rate"].max() + 0.12))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Cancellation Rate by Major Market Segment")
    ax.set_xlabel("Cancellation rate")
    ax.set_ylabel("Market segment")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path = IMAGE_DIR / "market_segment_cancellation_rate.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Created: {output_path.relative_to(PROJECT_ROOT)}")


def extract_png(notebook_path: Path, source_text: str, image_index: int) -> bytes:
    notebook = nbformat.read(notebook_path, as_version=4)
    matching_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and source_text in cell.source
    ]
    if len(matching_cells) != 1:
        raise ValueError(
            f"{notebook_path.name}: '{source_text}'に一致するセルが"
            f"{len(matching_cells)}件あります。"
        )

    png_outputs = [
        output.data["image/png"]
        for output in matching_cells[0].get("outputs", [])
        if output.output_type == "display_data" and "image/png" in output.data
    ]
    if image_index >= len(png_outputs):
        raise IndexError(
            f"{notebook_path.name}: PNG出力{image_index}が見つかりません。"
        )
    return base64.b64decode(png_outputs[image_index])


def main() -> None:
    IMAGE_DIR.mkdir(exist_ok=True)
    for spec in FIGURE_SPECS:
        png = extract_png(
            NOTEBOOK_DIR / spec["notebook"],
            spec["source_text"],
            spec["image_index"],
        )
        output_path = IMAGE_DIR / spec["filename"]
        output_path.write_bytes(png)
        print(f"Exported: {output_path.relative_to(PROJECT_ROOT)}")
    create_market_segment_figure()


if __name__ == "__main__":
    main()

