"""Plot supplied Work 1 control, prediction and planning-budget summaries."""

import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(input_dir, output_dir):
    TABLES = input_dir
    DEST = output_dir

    def csvrows(p):
        return list(csv.DictReader(p.open()))

    # Native plots, reading retained means and intervals; no bootstrap/model reruns.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#f6f2e8",
            "axes.edgecolor": "#274753",
            "text.color": "#274753",
            "axes.labelcolor": "#274753",
            "xtick.color": "#274753",
            "ytick.color": "#274753",
            "grid.color": "#ded7c8",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    models = ["diffusion", "autoregressive", "cvae", "flow_matching"]
    labels = ["Diffusion", "AR", "Joint CVAE", "Flow"]
    colors = dict(zip(models, ["#274753", "#8ab07c", "#299d8f", "#287779"]))
    control = csvrows(TABLES / "control.csv")
    pred = csvrows(TABLES / "prediction.csv")
    curves = csvrows(TABLES / "curves.csv")
    if any(
        not 0
        <= float(r["ci95_low_mm"])
        <= float(r["mean_euclidean_position_error_mm"])
        <= float(r["ci95_high_mm"])
        <= 56
        for r in pred
    ):
        raise ValueError(
            "Prediction intervals exceed the retained publication axes; adjust the layout explicitly"
        )

    def save(fig, name):
        fig.savefig(DEST / (name + ".pdf"))
        fig.savefig(DEST / (name + ".png"), dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.35, top=0.88, wspace=0.42)
    settings = [
        ("zero_lateral_correct", "0° lateral, correct", "#8ab07c", ""),
        ("all_correct", "All tilts, correct", "#299d8f", "//"),
        ("all_wrong", "All tilts, wrong", "#274753", ".."),
    ]
    width = 0.24
    for j, (group, label, color, hatch) in enumerate(settings):
        rows = [
            next(r for r in control if r["model"] == m and r["group"] == group)
            for m in models
        ]
        y = np.array([float(r["success_percent"]) for r in rows])
        lo = np.array([float(r["ci95_low_percent"]) for r in rows])
        hi = np.array([float(r["ci95_high_percent"]) for r in rows])
        x = np.arange(4) + (j - 1) * width
        axes[0].bar(
            x,
            y,
            width * 0.92,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
        axes[0].errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            fmt="none",
            color="#274753",
            elinewidth=0.85,
            capsize=2,
            zorder=4,
        )
    axes[0].set_xticks(np.arange(4), labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 110)
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].set_ylabel("Success (%)")
    axes[0].set_title("a  Proposal control", loc="left", fontweight="bold")
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(-0.08, -0.26),
        frameon=False,
        ncol=1,
        handlelength=1.5,
    )
    for model, label, marker, style in zip(
        models, labels, ["o", "s", "^", "D"], ["-", "--", "-.", ":"]
    ):
        rows = [r for r in pred if r["model"] == model]
        x = [float(r["prediction_step"]) for r in rows]
        y = [float(r["mean_euclidean_position_error_mm"]) for r in rows]
        axes[1].fill_between(
            x,
            [float(r["ci95_low_mm"]) for r in rows],
            [float(r["ci95_high_mm"]) for r in rows],
            color=colors[model],
            alpha=0.16,
            linewidth=0,
        )
        axes[1].plot(
            x,
            y,
            label=label,
            color=colors[model],
            marker=marker,
            linestyle=style,
            lw=1.6,
            ms=3,
        )
    axes[1].set_title("b  Imposed-action prediction", loc="left", fontweight="bold")
    axes[1].set_ylabel("Mean position error (mm)")
    axes[1].set_xlabel("Prediction step")
    axes[1].set_xticks(range(1, 7))
    axes[1].set_ylim(0, 56)
    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(-0.1, -0.26),
        frameon=False,
        ncol=2,
        handlelength=1.7,
        columnspacing=0.8,
    )
    for ax in axes:
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
    save(fig, "figure_2")
    cc = {
        "CEM": "#3f4349",
        "Proposal model": "#138a80",
        "Gated proposal model": "#d18419",
        "Large-K CEM": "#777777",
        "CVAE": "#3d65a5",
    }
    markers = {
        "CEM": "o",
        "Proposal model": "s",
        "Gated proposal model": "^",
        "Large-K CEM": "D",
        "CVAE": "P",
    }
    for panels, name in [("AB", "figure_a1_ab")]:
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.35))
        fig.subplots_adjust(left=0.095, right=0.98, bottom=0.28, top=0.88, wspace=0.28)
        for panel, ax in zip(panels, axes):
            series = list(
                dict.fromkeys(r["series"] for r in curves if r["panel"] == panel)
            )
            for series_name in series:
                rows = [
                    r
                    for r in curves
                    if r["panel"] == panel and r["series"] == series_name
                ]
                ax.plot(
                    [float(r["x"]) for r in rows],
                    [float(r["success_percent"]) for r in rows],
                    label=series_name,
                    color=cc[series_name],
                    marker=markers[series_name],
                    lw=1.5,
                    ms=4,
                )
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", lw=0.5)
            ax.set_axisbelow(True)
            if panel == "A":
                ax.set_xscale("log", base=2)
                ax.set_xticks(
                    [8, 32, 128, 512, 1024], ["8", "32", "128", "512", "1024"]
                )
                ax.set_xlabel("Candidates (log scale)")
                ax.set_title("A  Candidate budget", loc="left", fontweight="bold")
            elif panel == "B":
                ax.set_xscale("log")
                ax.set_xticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
                ax.set_xlabel("Time / decision (s, log scale)")
                ax.set_title("B  Measured planning time", loc="left", fontweight="bold")
        axes[0].set_ylabel("Success (%)")
        handles, labs = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labs,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.015),
            ncol=2 if panels == "AB" else 3,
            frameon=False,
            columnspacing=2,
            handlelength=2,
        )
        save(fig, name)
