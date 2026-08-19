"""Render the classification-flow diagram used by attribution.ipynb."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLOR_PRIMARY = "#2E5FA3"
COLOR_FLAG = "#C0392B"
COLOR_MUTED = "#9AA5B1"


def render(output_path=None):
    """Create the model-flow diagram and save it as a PNG."""
    output_path = Path(output_path or Path(__file__).with_name("classification_flow.png"))

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def add_box(x, y0, width, height, text, face, edge=COLOR_PRIMARY, fontsize=10):
        patch = FancyBboxPatch(
            (x, y0),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            linewidth=1.5,
            facecolor=face,
            edgecolor=edge,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y0 + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    def add_arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.4,
                color=COLOR_MUTED,
            )
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.035,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#45515C",
            )

    add_box(
        0.02,
        0.36,
        0.17,
        0.28,
        "One Task 1 candidate\n\n60 observable features\n(one row of X)",
        "#EAF1F8",
    )
    add_box(
        0.02,
        0.75,
        0.17,
        0.15,
        "Training only\n7 binary target columns",
        "#F7ECEA",
        edge=COLOR_FLAG,
        fontsize=9.5,
    )
    add_box(
        0.30,
        0.46,
        0.31,
        0.35,
        "Five independent logistic pipelines\n\n"
        "median imputation → standardization\n"
        "→ L2 logistic regression\n\n"
        "one target y_c per pipeline",
        "#EEF5EC",
        edge="#4D7C57",
    )
    add_box(
        0.30,
        0.12,
        0.31,
        0.20,
        "Two expert-signature rules\n\n"
        "mix shift + external demand\n"
        "(one event group each)",
        "#FFF5E6",
        edge="#B7791F",
    )
    add_box(
        0.76,
        0.28,
        0.21,
        0.44,
        "Seven independent scores\n\n"
        "score_genuine_efficiency\n"
        "score_spend_reduction\n"
        "score_survivorship\n"
        "score_creative_refresh\n"
        "score_measurement\n"
        "score_mix_shift\n"
        "score_external_demand",
        "#F3F0FA",
        edge="#6B5AA6",
        fontsize=9.3,
    )

    add_arrow(0.19, 0.50, 0.30, 0.63, "same X")
    add_arrow(0.19, 0.47, 0.30, 0.22)
    add_arrow(0.19, 0.82, 0.30, 0.75, "separate y_c")
    add_arrow(0.61, 0.63, 0.76, 0.58)
    add_arrow(0.61, 0.22, 0.76, 0.39)

    ax.text(
        0.5,
        0.97,
        "How one investigation becomes seven cause scores",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.865,
        0.18,
        "Scores do not compete\nand do not sum to 1",
        ha="center",
        va="top",
        fontsize=10,
        color="#6B5AA6",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(render())
