"""Analytic/model illustrations: Figure 1 and Supplement Figure S1."""

import argparse
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch


def analytic(out):
    u, v, d, sigma = 0.0, 0.5, 0.05, 0.05
    t = np.linspace(u, v, 501)
    x = np.linspace(-0.2, 0.75, 1001)
    density = lambda y: (ndtr((y - u) / sigma) - ndtr((y - v) / sigma)) / (v - u)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for values, label, style, color in [
        (t, "Truth", "-", "black"),
        (t + d, "Candidate P", "--", ".35"),
        (u + v - t, "Candidate Q", "-.", ".65"),
    ]:
        ax[0].plot(t, values, label=label, ls=style, color=color, lw=1.8)
    ax[1].plot(x, density(x), color="black", lw=2, label="Truth")
    ax[1].plot(x, density(x - d), color=".35", ls="--", lw=1.8, label="Candidate P")
    ax[1].plot(x, density(x), color=".65", ls="-.", lw=1.8, label="Candidate Q")
    ax[0].set(
        xlabel="State coordinate t",
        ylabel="Function value",
        title="(a) Functions at fixed locations",
    )
    ax[1].set(xlabel="Expression x", ylabel="Density", title="(b) Marginal expression densities")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=8)
    fig.savefig(out / "figure_S1_conditional_vs_marginal.pdf")
    fig.savefig(out / "figure_S1_conditional_vs_marginal.png", dpi=200)
    plt.close(fig)


def model(out):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.3, 0.3), 9.3, 5.3, fill=False, lw=1.2))
    ax.text(9.35, 0.5, r"$i=1,\ldots,n$", ha="right")
    ax.add_patch(Rectangle((4.2, 2.25), 4.9, 2.9, fill=False, lw=1.1))
    ax.text(8.9, 2.4, r"$c=1,\ldots,C_i$", ha="right")
    nodes = {
        "Z": (1.25, 1.55),
        "alpha": (3.3, 1.55),
        "beta": (6.55, 1.2),
        "S": (5.1, 4.4),
        "t": (7.5, 4.4),
        "X": (5.1, 3.1),
        "Y": (8.25, 3.1),
    }
    labels = {
        "Z": r"$Z_i$",
        "alpha": r"$\alpha_i$",
        "beta": r"$\beta_i$",
        "S": r"$S_{ic}$",
        "t": r"$t_{ic}$",
        "X": r"$X_{ic}$",
        "Y": r"$Y_{ic}$",
    }
    for key, xy in nodes.items():
        patch = (
            Rectangle(
                (xy[0] - 0.28, xy[1] - 0.28), 0.56, 0.56, facecolor="white", edgecolor="black"
            )
            if key in ("Z", "S", "X", "Y")
            else Circle(xy, 0.33, facecolor="white", edgecolor="black")
        )
        ax.add_patch(patch)
        ax.text(*xy, labels[key], ha="center", va="center", fontsize=13)
    for a, b, label, rad, dashed in [
        ("Z", "alpha", "A", 0, False),
        ("alpha", "beta", "B", 0, False),
        ("Z", "beta", "C", -0.35, False),
        ("S", "t", "support", 0, True),
        ("alpha", "X", r"$g_i$", 0, False),
        ("beta", "Y", r"$h_i$", 0, False),
        ("t", "X", "", 0, False),
        ("t", "Y", "", 0, False),
    ]:
        xy1, xy2 = nodes[a], nodes[b]
        ax.add_patch(
            FancyArrowPatch(
                xy1,
                xy2,
                arrowstyle="-|>",
                mutation_scale=13,
                shrinkA=20,
                shrinkB=20,
                connectionstyle=f"arc3,rad={rad}",
                lw=1.2,
                linestyle="--" if dashed else "-",
            )
        )
        if label:
            x, y = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
            if label == "C":
                y = 0.55
            else:
                y += 0.22
            ax.text(
                x,
                y,
                label,
                ha="center",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
            )
    fig.savefig(out / "figure_1_model.pdf", bbox_inches="tight")
    fig.savefig(out / "figure_1_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    model(a.output)
    analytic(a.output)
