import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

input_dir = Path("convergence_data/cora")
output_dir = input_dir / "plots_stopping"
output_dir.mkdir(parents=True, exist_ok=True)

# Les zéros sont affichés à cette valeur uniquement pour le tracé.
plot_floor = 1e-12

for trial in range(1, 11):
    filename = input_dir / f"trial_init{trial}.json"

    with filename.open("r", encoding="utf-8") as f:
        data = json.load(f)

    errors = np.asarray(data["error"], dtype=float)
    nmi = np.asarray(data["NMI"], dtype=float)

    if len(errors) != len(nmi):
        raise ValueError(
            f"{filename.name}: error and NMI must have the same length."
        )

    error_diff = np.abs(np.diff(errors))
    nmi_diff = np.abs(np.diff(nmi))

    iterations = np.arange(1, len(errors))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        iterations, np.maximum(error_diff, plot_floor),
        marker="o", markersize=3,
        label=r"$|e_t - e_{t-1}|$"
    )
    ax.plot(
        iterations, np.maximum(nmi_diff, plot_floor),
        marker="s", markersize=3,
        label=r"$|\mathrm{NMI}_t - \mathrm{NMI}_{t-1}|$"
    )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Absolute change (log scale)")
    ax.set_title(f"Initialization {trial}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / f"stopping_trial{trial}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)