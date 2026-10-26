"""Plot executed forecast-selection on/off effects and cumulative costs."""

import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CASES = tuple(f"stock_test_p{i:03}" for i in range(6))


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_data(source):
    paired = read_rows(source / "paired.csv")
    traces = read_rows(source / "cumulative_cost.csv")
    if tuple(r["case_id"] for r in paired) != CASES or len(traces) != 8064:
        raise AssertionError(
            "All six fixed cases and all8064 cumulative records required"
        )
    for row in paired:
        for key in ("on_cost", "off_cost", "on_vs_off_percent"):
            row[key] = float(row[key])
        expected = 100.0 * (row["on_cost"] / row["off_cost"] - 1.0)
        if not np.isclose(row["on_vs_off_percent"], expected, atol=1e-10, rtol=1e-10):
            raise AssertionError("Paired percentage does not match native costs")
    by_case = {}
    for case in CASES:
        rows = [r for r in traces if r["case_id"] == case]
        days = np.array([float(r["day"]) for r in rows])
        values = np.array([float(r["on_vs_off_cumulative_percent"]) for r in rows])
        on = np.array([float(r["on_integrated_cost"]) for r in rows])
        off = np.array([float(r["off_integrated_cost"]) for r in rows])
        if len(rows) != 1344 or not np.allclose(
            days, np.arange(1, 1345) / 96.0, atol=1e-10, rtol=0
        ):
            raise AssertionError("Exactly1344 actual15-minute points per case required")
        if not np.isfinite(np.r_[days, values, on, off]).all() or not (off > 0).all():
            raise AssertionError(
                "Finite physical cumulative values and positive denominator required"
            )
        if not np.allclose(values, 100.0 * (on / off - 1.0), atol=1e-10, rtol=1e-10):
            raise AssertionError("Cumulative percentage arithmetic failed")
        final = next(r["on_vs_off_percent"] for r in paired if r["case_id"] == case)
        if not np.isclose(values[-1], final, atol=1e-8, rtol=1e-8):
            raise AssertionError(
                "Cumulative endpoint must match the full-episode effect"
            )
        by_case[case] = dict(day=days, value=values)
    return paired, by_case


def render(input_dir, output_dir):
    source = input_dir
    destination = output_dir
    paired, by_case = read_data(source)
    dark, teal, grey, grid = "#274753", "#299D8F", "#8F8A7D", "#DED7C8"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "text.color": dark,
            "axes.labelcolor": dark,
            "xtick.color": dark,
            "ytick.color": dark,
            "axes.edgecolor": grey,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )
    values = np.array([r["on_vs_off_percent"] for r in paired])
    mean = float(values.mean())
    labels = [
        f"{r['weather'].capitalize()} · {r['case_id'].split('_')[-1]}" for r in paired
    ] + ["Equal-case mean"]
    ys = np.array([6, 5, 4, 3, 2, 1, -0.35])
    all_values = np.r_[values, mean]
    low, high = min(0.0, float(all_values.min())), max(0.0, float(all_values.max()))
    span = max(high - low, 0.05)
    fig, ax = plt.subplots(figsize=(468 / 72.0, 218 / 72.0))
    fig.subplots_adjust(left=0.235, right=0.93, bottom=0.23, top=0.96)
    ax.set_facecolor("#F6F2E8")
    ax.axvline(0, color=grey, lw=1.0, zorder=1)
    ax.axhline(0.35, color=grid, lw=0.7, zorder=1)
    ax.scatter(values, ys[:6], color=dark, s=29, marker="o", zorder=3)
    ax.scatter(
        [mean],
        [ys[-1]],
        color=teal,
        edgecolors=dark,
        linewidths=0.6,
        s=55,
        marker="D",
        zorder=3,
    )
    for x, y in zip(all_values, ys):
        ax.annotate(
            f"{x:+.3f}%",
            (x, y),
            xytext=(6 if x >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=8.5,
            fontweight="bold" if y < 0 else "normal",
        )
    ax.set_xlim(low - 0.24 * span, high + 0.24 * span)
    ax.set_ylim(-0.95, 6.55)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", color=grid, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "Economic cost change: forecast scoring on versus off (%)", labelpad=7
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    for ext in ("pdf", "png", "svg"):
        path = destination / f"forecast_control.{ext}"
        fig.savefig(path, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 3, figsize=(468 / 72.0, 285 / 72.0), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.115, right=0.98, bottom=0.16, top=0.92, wspace=0.16, hspace=0.32
    )
    extent = max(max(abs(r["value"])) for r in by_case.values())
    extent = max(extent * 1.1, 0.05)
    for ax, row in zip(axes.flat, paired):
        data = by_case[row["case_id"]]
        # The first point is the first saved observation, never an invented(0,0).
        x, y = data["day"], data["value"]
        ax.axhline(0.0, color=grey, lw=0.8)
        ax.plot(x, y, color=dark, lw=1.0)
        ax.scatter([x[-1]], [y[-1]], s=14, color=dark, zorder=3, clip_on=False)
        ax.set_title(
            f"{row['weather'].capitalize()} · {row['case_id'].split('_')[-1]}",
            loc="left",
            pad=5,
        )
        ax.set_xlim(0, 14)
        ax.set_ylim(-extent, extent)
        ax.set_xticks([0, 7, 14])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(color=grid, linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
    fig.supxlabel("Elapsed days", fontsize=9, y=0.035)
    fig.supylabel(
        "Cumulative economic cost change: on versus off (%)", fontsize=9, x=0.012
    )
    for ext in ("pdf", "png", "svg"):
        path = destination / f"forecast_control_traces.{ext}"
        fig.savefig(path, dpi=220)
    plt.close(fig)
