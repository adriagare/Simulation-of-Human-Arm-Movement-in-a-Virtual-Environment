"""
Genera figures il·lustratives de trajectòries reach per a la memòria.

Per cada participant seleccionat, dibuixa:
  * Vista superior (XZ) de la trajectoria de la ma REAL i de la ma MOSTRADA
    durant trials de l'angle de control (nivell 1, 0 graus) i d'angle alt
    (nivell 10, 72 graus aprox).
  * Perfil de velocitat de la ma real per als mateixos trials.

Ús:
    python trajectory_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data_analysis" / "aggregate"


def plot_trajectory_comparison(folder: Path, out_path: Path,
                               low_level: int = 1, high_level: int = 10):
    """Per al participant a `folder`, dibuixa trajectories de comparacio."""
    pid = folder.name.split("_")[0]
    exp_files = list(folder.glob("experiment_*.csv"))
    if not exp_files:
        print(f"[{pid}] cap experiment_*.csv")
        return
    df = pd.read_csv(exp_files[0])

    probe = df[df["phase"] == "ProbeActive"].copy()
    low_trials  = probe[probe["probe_level"] == low_level]
    high_trials = probe[probe["probe_level"] == high_level]

    if low_trials.empty or high_trials.empty:
        print(f"[{pid}] no trials suficients als nivells {low_level}/{high_level}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Participant {pid} - comparacio reach control vs reach a angle alt",
                 fontsize=14, fontweight="bold")

    # --- Panells superiors: vista superior (XZ) ---
    for col, (level, trials, title) in enumerate([
        (low_level,  low_trials,  f"Control (nivell {low_level}, ~0 graus)"),
        (high_level, high_trials, f"Pertorbacio alta (nivell {high_level}, ~72 graus held)"),
    ]):
        ax = axes[0, col]
        cmap = plt.get_cmap("viridis")
        n_trials = trials["probe_trial"].nunique()
        for i, (trl, grp) in enumerate(trials.groupby("probe_trial")):
            if trl > 5:
                continue
            color = cmap(i / max(1, n_trials - 1))
            # Ma real
            ax.plot(grp["hand_real_x"], grp["hand_real_z"],
                    "-", color=color, alpha=0.8, lw=1.5,
                    label=f"trial {int(trl)} real" if i == 0 else None)
            # Ma mostrada (despres de la pertorbacio)
            ax.plot(grp["hand_unity_x"], grp["hand_unity_z"],
                    "--", color=color, alpha=0.6, lw=1.0,
                    label=f"trial {int(trl)} mostrada" if i == 0 else None)
            # Posicio creu
            cx = grp["cross_x"].iloc[0]
            cz = grp["cross_z"].iloc[0]
            ax.plot(cx, cz, "x", color=color, markersize=12, mew=2)
        ax.set_xlabel("X (m)  [costat esquerre - costat dret]")
        ax.set_ylabel("Z (m)  [a prop - lluny]")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        if col == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- Panells inferiors: perfil de velocitat (ma real) ---
    for col, (level, trials, title) in enumerate([
        (low_level,  low_trials,  f"Perfil velocitat control"),
        (high_level, high_trials, f"Perfil velocitat alta"),
    ]):
        ax = axes[1, col]
        cmap = plt.get_cmap("viridis")
        n_trials = trials["probe_trial"].nunique()
        for i, (trl, grp) in enumerate(trials.groupby("probe_trial")):
            if trl > 5:
                continue
            color = cmap(i / max(1, n_trials - 1))
            t = grp["timestamp"].to_numpy()
            t = t - t[0]  # alinear a 0
            hx = grp["hand_real_x"].to_numpy()
            hy = grp["hand_real_y"].to_numpy()
            hz = grp["hand_real_z"].to_numpy()
            dt = np.diff(t)
            dt[dt <= 0] = 1e-3
            speed = np.sqrt(np.diff(hx)**2 + np.diff(hy)**2 + np.diff(hz)**2) / dt
            ax.plot(t[1:], speed, "-", color=color, alpha=0.8, lw=1.2,
                    label=f"trial {int(trl)}")
        ax.set_xlabel("Temps des de l'inici del trial (s)")
        ax.set_ylabel("Velocitat ma real (m/s)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[{pid}] desat: {out_path}")


def plot_displaced_hand_overlay(folder: Path, out_path: Path, high_level: int = 10):
    """Vista superior detall: ma real vs ma mostrada en trials d'angle alt,
    superposant els cinc trials del nivell per mostrar la magnitud de la dissociacio."""
    pid = folder.name.split("_")[0]
    exp_files = list(folder.glob("experiment_*.csv"))
    if not exp_files:
        return
    df = pd.read_csv(exp_files[0])
    probe = df[df["phase"] == "ProbeActive"]
    trials = probe[probe["probe_level"] == high_level]
    if trials.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(f"Participant {pid} - dissociacio ma real vs ma mostrada (nivell {high_level})",
                 fontsize=13, fontweight="bold")
    cmap = plt.get_cmap("plasma")
    for i, (trl, grp) in enumerate(trials.groupby("probe_trial")):
        if trl > 5:
            continue
        col = cmap(i / 5)
        ax.plot(grp["hand_real_x"], grp["hand_real_z"], "-", color=col, alpha=0.9, lw=1.8)
        ax.plot(grp["hand_unity_x"], grp["hand_unity_z"], "--", color=col, alpha=0.6, lw=1.3)
        # Posicio creu
        cx = grp["cross_x"].iloc[0]
        cz = grp["cross_z"].iloc[0]
        ax.plot(cx, cz, "x", color=col, markersize=14, mew=2.5)
        # Posicio elbow al final
        ex = grp["elbow_real_x"].iloc[-1]
        ez = grp["elbow_real_z"].iloc[-1]
        ax.plot(ex, ez, "o", color=col, markersize=7, alpha=0.7,
                markerfacecolor="white", markeredgewidth=1.5)
    # Llegenda artificial
    ax.plot([], [], "-", color="grey", lw=1.8, label="Ma REAL (la que mou el subjecte)")
    ax.plot([], [], "--", color="grey", lw=1.3, label="Ma MOSTRADA al casc (rotada)")
    ax.plot([], [], "x", color="grey", markersize=12, mew=2, label="Creu objectiu")
    ax.plot([], [], "o", color="grey", markersize=7, markerfacecolor="white", markeredgewidth=1.5,
            label="Colze final (pivot)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[{pid}] desat overlay: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Triem un participant representatiu: P03 (Anna) - 11 nivells complets,
    # llindar cinematic 16 graus i verbal 72 graus, perfil tipic
    target_pid = "P03"
    target_folder = None
    for f in sorted((PROJECT_ROOT / "exp").iterdir()):
        if f.is_dir() and f.name.startswith(target_pid):
            target_folder = f; break
    if target_folder is None:
        print(f"[ERROR] participant {target_pid} no trobat")
        return
    plot_trajectory_comparison(target_folder,
                               OUT_DIR / "trajectory_comparison.png",
                               low_level=1, high_level=10)
    plot_displaced_hand_overlay(target_folder,
                                OUT_DIR / "displaced_hand_overlay.png",
                                high_level=10)


if __name__ == "__main__":
    main()
