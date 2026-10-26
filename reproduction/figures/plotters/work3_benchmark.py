"""Plot complete paired benchmark outcomes and mean controller times."""

import csv
import math
from statistics import mean
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(input_dir, output_dir):
    ORDER = [
        "genaif_selective_mean",
        "public_history_h1",
        "oracle_selected",
        "stock_han",
        "stock_yang",
        "expert_pi",
        "public_pi",
    ]
    LABELS = [
        "GenAIF",
        "Encoder-conditioned\ndiffusion",
        "Truth-conditioned\ndiffusion",
        "Han Koopman",
        "Yang offline RL",
        "Expert PI",
        "Native PI",
    ]
    WIDTH, HEIGHT = 468, 266

    palette = {
        "controllers": {
            "genaif_selective_mean": {
                "label": "GenAIF",
                "color": "#274753",
                "marker": "D",
                "linestyle": "-",
                "filled": True,
            },
            "public_history_h1": {
                "label": "Encoder-conditioned diffusion",
                "color": "#299D8F",
                "marker": "s",
                "linestyle": "-.",
                "filled": True,
            },
            "oracle_selected": {
                "label": "Truth-conditioned diffusion",
                "color": "#BF7628",
                "marker": "^",
                "linestyle": ":",
                "filled": True,
            },
            "stock_han": {
                "label": "Han Koopman",
                "color": "#8AB07C",
                "marker": "D",
                "linestyle": "--",
                "filled": False,
            },
            "stock_yang": {
                "label": "Yang offline RL",
                "color": "#8AB07C",
                "marker": "^",
                "linestyle": "-.",
                "filled": True,
            },
            "expert_pi": {
                "label": "Expert PI",
                "color": "#287779",
                "marker": "*",
                "linestyle": "-",
                "filled": True,
            },
            "public_pi": {
                "label": "Native PI",
                "color": "#8F8A7D",
                "marker": "o",
                "linestyle": ":",
                "filled": False,
            },
        },
        "ink": "#274753",
        "grid": "#DED7C8",
        "panel_background": "#F6F2E8",
        "reference_line": "#8F8A7D",
    }
    styles = palette["controllers"]
    with (input_dir / "benchmark.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["case_id"], row["method"]): row for row in rows}
    cases = sorted({row["case_id"] for row in rows})
    assert len(rows) == len(lookup) == 42
    assert len(cases) == 6
    assert {row["method"] for row in rows} == set(ORDER)
    assert cases == [f"stock_test_p{i:03d}" for i in range(6)]

    plot_rows = []
    for method in ORDER:
        method_rows = [lookup[case, method] for case in cases]
        values = []
        for case, row in zip(cases, method_rows):
            cost = float(row["cost_per_day"])
            pi = float(lookup[case, "public_pi"]["cost_per_day"])
            value = 100 * (cost / pi - 1)
            assert math.isclose(
                float(row["eqi"]) + float(row["weighted_oci"]), cost, rel_tol=1e-12
            )
            assert math.isclose(
                float(row["weighted_oci"]), 0.3 * float(row["oci"]), rel_tol=1e-12
            )
            assert math.isclose(float(row["integrated_cost"]), 14 * cost, rel_tol=1e-12)
            values.append(value)
        if any(not -2.4 <= value <= 10.9 for value in values):
            raise ValueError(
                "A cost change exceeds the retained publication axes; adjust the layout explicitly"
            )
        cost_mean = mean(values)
        time_mean = mean(
            float(row["decision_mean_seconds"]) * 1000 for row in method_rows
        )
        plot_rows.append(
            {
                "method": method,
                "display_label": styles[method]["label"],
                "cases": cases,
                "paired_cost_percent": values,
                "mean_paired_cost_percent": cost_mean,
                "mean_controller_ms": time_mean,
                "display_cost": f"{cost_mean:+.3f}%",
                "display_time": f"{time_mean:.3f}",
                "color": styles[method]["color"],
                "marker": styles[method]["marker"],
                "filled": styles[method]["filled"],
            }
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "text.color": palette["ink"],
            "axes.labelcolor": palette["ink"],
            "xtick.color": palette["ink"],
            "ytick.color": palette["ink"],
            "axes.edgecolor": palette["grid"],
            "axes.facecolor": palette["panel_background"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=144)
    left, bottom, plot_width, plot_height = 117, 35, 231, 203
    ax = fig.add_axes(
        [left / WIDTH, bottom / HEIGHT, plot_width / WIDTH, plot_height / HEIGHT]
    )
    ax.set_xlim(-2.4, 10.9)
    ax.set_ylim(6.6, -0.6)
    ax.set_xticks([-2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks(range(7), LABELS, fontsize=8.4)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=8, length=2.5, width=0.5, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=palette["grid"], alpha=0.85, linewidth=0.45)
    ax.axvline(0, color=palette["reference_line"], linewidth=0.9, zorder=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)

    case_collections = []
    mean_collections = []
    offsets = np.linspace(-0.18, 0.18, 6)
    for index, row in enumerate(plot_rows):
        style = styles[row["method"]]
        case_collections.append(
            ax.scatter(
                row["paired_cost_percent"],
                index + offsets,
                s=13.5,
                marker=style["marker"],
                edgecolors=style["color"],
                facecolors=style["color"]
                if style["filled"]
                else palette["panel_background"],
                linewidths=0.8,
                alpha=0.60,
                zorder=3,
            )
        )
        mean_collections.append(
            ax.scatter(
                row["mean_paired_cost_percent"],
                index,
                s=43,
                marker="D",
                color=style["color"],
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
        )
        for x_pt, text, bold in [
            (381, row["display_cost"], True),
            (440, row["display_time"], False),
        ]:
            ax.text(
                (x_pt - left) / plot_width,
                index,
                text,
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="center",
                fontsize=8.5,
                fontweight="bold" if bold else "normal",
                clip_on=False,
            )
    ax.set_xlabel("Economic cost change versus PI (%)", fontsize=8.5, labelpad=5)
    fig.text(
        381 / WIDTH,
        249 / HEIGHT,
        "Mean change\n(%)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        linespacing=1.12,
    )
    fig.text(
        440 / WIDTH,
        249 / HEIGHT,
        "Controller\ntime (ms)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        linespacing=1.12,
    )

    fig.savefig(output_dir / "benchmark.pdf")
    fig.savefig(output_dir / "benchmark.png", dpi=240)
    plt.close(fig)
