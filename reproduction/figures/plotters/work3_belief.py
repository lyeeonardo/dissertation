"""Plot the first two hours of saved storm-startup particles and weights."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


def render(input_dir, output_dir):
    INK = "#274753"
    GRID = "#DED7C8"
    PANEL = "#F6F2E8"
    REFERENCE = "#8F8A7D"
    z = {
        k: np.array(v)
        for k, v in json.loads((input_dir / "belief_startup.json").read_text()).items()
    }
    x, w = z["particles"], z["weights"]
    assert x.shape == (9, 5, 8, 2) and w.shape == (9, 5, 8)
    assert np.allclose(w.sum(-1), 1, atol=1e-14)
    if not (
        np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(z["truth"]).all()
    ):
        raise ValueError("Belief inputs must be finite")
    if not ((w[:, 1] * 100 >= 11.0).all() and (w[:, 1] * 100 <= 13.5).all()):
        raise ValueError(
            "Particle weights exceed the retained expanded axes; adjust the layout explicitly"
        )
    assert np.array_equal(w[:, 0], w[:, 1])
    assert np.array_equal(x[:, 3], x[:, 4]) and np.array_equal(w[:, 3], w[:, 4])
    assert np.array_equal(x[1:, 0], x[:-1, 3])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": REFERENCE,
            "axes.linewidth": 0.55,
            "axes.labelcolor": INK,
            "axes.facecolor": PANEL,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.facecolor": "#FFFFFF",
        }
    )
    # Particle identities, not model-family semantics. Reuse the published
    # navy/teal/sage/ochre palette without altering any identity or trajectory.
    PARTICLE_TONES = {"slate": "#5E8491", "dark_sage": "#587A48"}
    COLORS = [
        "#274753",
        PARTICLE_TONES["slate"],
        "#BF7628",
        "#8AB07C",
        "#287779",
        "#BA902C",
        "#299D8F",
        PARTICLE_TONES["dark_sage"],
    ]
    fig = plt.figure(figsize=(468 / 72, 334 / 72), facecolor="white")
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1, 1, 0.74],
        left=0.115,
        right=0.975,
        bottom=0.125,
        top=0.84,
        hspace=0.42,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    t = np.arange(9) / 4
    for ax, d, title, unit in [
        (axes[0], 0, "a   Dissolved oxygen · reactor 5", r"mg O$_2$/L"),
        (axes[1], 1, "b   Nitrate · reactor 2", r"mg N/L"),
    ]:
        for k, color in enumerate(COLORS):
            # Every propagation segment connects the prepared current state to its own
            # selected next-state forecast. Correction is a vertical same-boundary move.
            propagation = [
                [(t[j], float(x[j, 1, k, d])), (t[j + 1], float(x[j, 2, k, d]))]
                for j in range(8)
            ]
            corrections = [
                [(t[j], float(x[j, 0, k, d])), (t[j], float(x[j, 1, k, d]))]
                for j in range(1, 9)
            ]
            ax.add_collection(
                LineCollection(
                    propagation, colors=color, linewidths=0.85, alpha=0.88, zorder=3
                )
            )
            ax.add_collection(
                LineCollection(
                    corrections,
                    colors=color,
                    linewidths=0.7,
                    linestyles=(0, (1.6, 1.2)),
                    alpha=0.78,
                    zorder=3,
                )
            )
            ax.plot(
                t,
                x[:9, 1, k, d],
                linestyle="none",
                marker="o",
                ms=1.7,
                color=color,
                zorder=4,
            )
        ax.plot(t, z["truth"][:9, d], color=INK, ls=(0, (4, 2)), lw=1.2, zorder=2)
        ax.set_title(title, loc="left", pad=4, fontweight="semibold")
        ax.set_ylabel(unit, labelpad=5)
        vals = np.r_[x[:9, 1, :, d].ravel(), x[:8, 2, :, d].ravel(), z["truth"][:9, d]]
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.09
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, 2)
        ax.set_xticks(np.arange(0, 2.1, 0.5))
        ax.tick_params(axis="x", labelbottom=False, length=2.5)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=2.5)

    ax = axes[2]
    for k, color in enumerate(COLORS):
        ax.plot(
            t,
            w[:9, 1, k] * 100,
            color=color,
            lw=0.95,
            marker="o",
            markersize=1.7,
            zorder=3,
        )
    ax.axhline(12.5, color=REFERENCE, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(11.0, 13.5)
    ax.set_yticks([11.0, 12.0, 12.5, 13.0])
    ax.set_xlim(0, 2)
    ax.set_xticks(np.arange(0, 2.1, 0.5))
    ax.set_xlabel("Time from startup (hours)", labelpad=4)
    ax.set_ylabel("Belief mass (%)", labelpad=5)
    ax.set_title(
        "c   Particle weights · expanded scale around 12.5%",
        loc="left",
        pad=4,
        fontweight="semibold",
    )
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    fig.text(
        0.115,
        0.976,
        "Final controller · storm startup · first 2 hours · detail",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="semibold",
    )
    handles = [
        Line2D([0], [0], color=c, lw=1.5, marker="o", ms=2.5, label=str(i + 1))
        for i, c in enumerate(COLORS)
    ]
    fig.text(0.115, 0.919, "Particle", ha="left", va="center", fontsize=8.5)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.215, 0.919),
        ncol=8,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.35,
        borderaxespad=0,
    )
    semantics = [
        Line2D([0], [0], color=REFERENCE, lw=1, label="Forecast propagation"),
        Line2D(
            [0], [0], color=REFERENCE, lw=1, ls=(0, (1.6, 1.2)), label="Mean correction"
        ),
        Line2D(
            [0], [0], color=INK, lw=1.2, ls=(0, (4, 2)), label="Truth (evaluation only)"
        ),
    ]
    fig.legend(
        handles=semantics,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.002),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=2.3,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(output_dir / f"belief_evolution_zoom.{suffix}", dpi=210)
    plt.close(fig)
