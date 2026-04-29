"""
Behavioural analysis of perturbation-detection probe data.

Reads experiment_*.csv (per-frame log) and (optionally) the matching
events_*.csv (sparse experimenter annotations) produced by
ExperimentController when `enableProbePhase = true`.

For every probe trial it computes objective kinematic metrics from the
real-hand trajectory and groups them by perturbation magnitude |θ|. No
binary self-report is required: the goal is to detect at which |θ| the
participant's natural reaching behaviour starts to deviate.

Metrics computed per trial:
    * movement_time          — duration from movement onset (10 % of peak
                               speed) to offset (back below 10 %).
    * num_submovements       — count of distinct velocity peaks above
                               20 % of trial peak speed, separated by a
                               drop below 50 % of that peak.
    * peak_speed             — max |velocity| during the trial.
    * time_to_peak_norm      — TPV normalised by movement time (0..1).
                               Natural reaches: ~0.4-0.5.
    * endpoint_dev           — final hand-cross distance (m).
    * path_length            — total travelled distance of the real hand.
    * straightness           — endpoint_dev / path_length proxy.
                               (Lower = more curved trajectories.)

Usage:
    python probe_analysis.py path/to/experiment_*.csv \\
                             [--events events_*.csv] [--save out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
#  Trial-level metrics
# ──────────────────────────────────────────────────────────────────────

def trial_metrics(trial: pd.DataFrame) -> dict:
    """Compute kinematic metrics for one probe trial."""
    if len(trial) < 5:
        return {}

    t  = trial["timestamp"].to_numpy(dtype=float)
    hx = trial["hand_unity_x"].to_numpy(dtype=float)
    hy = trial["hand_unity_y"].to_numpy(dtype=float)
    hz = trial["hand_unity_z"].to_numpy(dtype=float)

    # Per-frame speed magnitude (from logged velocity columns)
    vx = trial["vel_x"].to_numpy(dtype=float)
    vy = trial["vel_y"].to_numpy(dtype=float)
    vz = trial["vel_z"].to_numpy(dtype=float)
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)

    if speed.max() <= 1e-6:
        return {}

    peak = speed.max()
    onset_thr  = 0.10 * peak
    offset_thr = 0.10 * peak

    # Movement window
    above = speed > onset_thr
    if not above.any():
        return {}
    onset_idx  = int(np.argmax(above))
    last_above = int(len(above) - 1 - np.argmax(above[::-1]))
    move_t = float(t[last_above] - t[onset_idx])

    # Submovement count: peaks above 20 % of peak that re-emerge after
    # a drop below 50 % of peak.
    hi = 0.20 * peak
    lo = 0.50 * peak
    n_sub = 0
    state_above = False
    seen_below = True
    for s in speed[onset_idx:last_above + 1]:
        if not state_above and s > hi and seen_below:
            n_sub += 1
            state_above = True
            seen_below = False
        elif state_above and s < lo:
            state_above = False
            seen_below = True

    # Time to peak velocity (within movement window), normalised
    seg = speed[onset_idx:last_above + 1]
    tpv_idx = onset_idx + int(np.argmax(seg))
    tpv_norm = (t[tpv_idx] - t[onset_idx]) / max(move_t, 1e-3)

    # Endpoint deviation (last frame vs. cross position)
    cross = trial[["cross_x", "cross_y", "cross_z"]].iloc[-1].to_numpy(dtype=float)
    final = np.array([hx[-1], hy[-1], hz[-1]])
    endpoint_dev = float(np.linalg.norm(final - cross))

    # Path length and straightness
    dpos = np.diff(np.column_stack([hx, hy, hz]), axis=0)
    path_length = float(np.sum(np.linalg.norm(dpos, axis=1)))
    direct = float(np.linalg.norm(np.array([hx[-1] - hx[onset_idx],
                                            hy[-1] - hy[onset_idx],
                                            hz[-1] - hz[onset_idx]])))
    straightness = direct / max(path_length, 1e-3)

    return dict(
        movement_time=move_t,
        num_submovements=int(n_sub),
        peak_speed=float(peak),
        time_to_peak_norm=float(tpv_norm),
        endpoint_dev=endpoint_dev,
        path_length=path_length,
        straightness=straightness,
    )


# ──────────────────────────────────────────────────────────────────────
#  Loading and grouping
# ──────────────────────────────────────────────────────────────────────

def load_probe_metrics(csv_paths: list[Path]) -> pd.DataFrame:
    """Return one row per probe trial with its metrics + |theta|."""
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "probe_angle" not in df.columns:
            print(f"[warn] {p.name} has no probe_angle column — skipping.")
            continue
        # Probe trials = phase string contains 'Probe' AND probe_level > 0
        in_probe = df["phase"].astype(str).str.contains("Probe", na=False)
        sub = df[in_probe & (df["probe_level"] > 0)].copy()
        if sub.empty:
            continue

        for (lvl, trl), trial in sub.groupby(["probe_level", "probe_trial"]):
            metrics = trial_metrics(trial)
            if not metrics:
                continue
            rows.append(dict(
                source=p.name,
                probe_level=int(lvl),
                probe_trial=int(trl),
                probe_angle=float(trial["probe_angle"].iloc[0]),
                abs_theta=float(abs(trial["probe_angle"].iloc[0])),
                **metrics,
            ))

    if not rows:
        raise SystemExit("No probe trials found in the provided CSVs.")
    return pd.DataFrame(rows)


def load_events(paths: list[Path]) -> pd.DataFrame:
    out = []
    for p in paths:
        try:
            out.append(pd.read_csv(p))
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
    if not out:
        return pd.DataFrame(columns=["time", "trial", "phase", "probe_level",
                                     "probe_angle", "label"])
    return pd.concat(out, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_panels(df: pd.DataFrame, save: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    panels = [
        ("num_submovements",  "Sub-movements per trial",       "count"),
        ("movement_time",     "Movement time",                  "s"),
        ("endpoint_dev",      "Endpoint deviation (real hand vs cross)",  "m"),
        ("straightness",      "Trajectory straightness  (1 = perfectly straight)", ""),
    ]

    for ax, (col, title, unit) in zip(axes.flat, panels):
        groups = df.groupby("abs_theta")[col]
        thetas = sorted(df["abs_theta"].unique())
        data   = [groups.get_group(th).to_numpy() for th in thetas]
        ax.boxplot(data, labels=[f"{th:g}" for th in thetas])
        means = [np.mean(g) for g in data]
        ax.plot(range(1, len(thetas) + 1), means, "o-", c="C3", lw=1.5,
                label="mean")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("|θ|  (deg)")

    fig.suptitle("Behavioural metrics vs. perturbation magnitude", y=1.00)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"[info] figure saved to {save}")
    else:
        plt.show()


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("abs_theta")
              .agg(n=("movement_time", "size"),
                   mt_mean=("movement_time", "mean"),
                   mt_std=("movement_time", "std"),
                   nsub_mean=("num_submovements", "mean"),
                   ep_mean=("endpoint_dev", "mean"),
                   ep_std=("endpoint_dev", "std"),
                   straight_mean=("straightness", "mean"))
              .round(3)
              .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+", type=Path, help="experiment_*.csv files")
    ap.add_argument("--events", nargs="*", type=Path, default=[],
                    help="Optional events_*.csv files for spontaneous annotations.")
    ap.add_argument("--save", type=Path, default=None,
                    help="Save the figure instead of showing it.")
    args = ap.parse_args()

    df = load_probe_metrics(args.csvs)
    print(f"[info] {len(df)} probe trials parsed across {len(args.csvs)} file(s).\n")

    summary = summary_table(df)
    print("Trial-level summary by |θ|:\n")
    print(summary.to_string(index=False))

    if args.events:
        ev = load_events(args.events)
        if not ev.empty:
            print("\nSpontaneous events:")
            print(ev.sort_values(["trial", "time"]).to_string(index=False))

    plot_panels(df, args.save)


if __name__ == "__main__":
    main()
