"""
Behavioural analysis of perturbation-detection probe data.

Reads experiment_*.csv (per-frame log) and (optionally) the matching
events_*.csv (sparse experimenter annotations) produced by
ExperimentController during the probe phase of a session.

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

Outputs (one set per input CSV, written under <project_root>/data_analysis/):
    grafics/<stem>.png      — 4-panel boxplot vs. |θ|
    resultats/<stem>.json   — structured per-|θ| metrics + event/hit summary
    resultats/<stem>.txt    — human-readable digest of the same content

Usage:
    python probe_analysis.py path/to/experiment_*.csv \\
                             [--events events_*.csv] [--save out.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Hit detection radius used by ExperimentController (real-hand distance to cross).
DETECTION_RADIUS_M = 0.10

# Project root: probe_analysis.py lives at <root>/ml/, so parent.parent is root.
PROJECT_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis"

# The metrics we report and on which we estimate detection thresholds.
METRIC_NAMES = [
    "movement_time", "num_submovements", "peak_speed",
    "time_to_peak_norm", "endpoint_dev", "path_length", "straightness",
    "forearm_compensation_abs_deg", "forearm_compensation_aligned_deg",
]


# ──────────────────────────────────────────────────────────────────────
#  Trial-level metrics
# ──────────────────────────────────────────────────────────────────────

def trial_metrics(trial: pd.DataFrame) -> dict:
    """Compute kinematic metrics for one probe trial.

    Uses hand_real_* columns when present (the participant's marker-derived
    hand, which is the signal of interest: how the real arm reacts to the
    visual mismatch). Falls back to hand_unity_* for backward compatibility.
    """
    if len(trial) < 5:
        return {}

    has_real = {"hand_real_x", "hand_real_y", "hand_real_z"}.issubset(trial.columns)
    hx_col, hy_col, hz_col = (
        ("hand_real_x", "hand_real_y", "hand_real_z") if has_real
        else ("hand_unity_x", "hand_unity_y", "hand_unity_z")
    )

    t  = trial["timestamp"].to_numpy(dtype=float)
    hx = trial[hx_col].to_numpy(dtype=float)
    hy = trial[hy_col].to_numpy(dtype=float)
    hz = trial[hz_col].to_numpy(dtype=float)

    # Speed from finite differences of the chosen hand stream. The logged
    # vel_* columns track the displayed hand, which during probe trials is
    # offset from the real hand by the perturbation envelope; recomputing
    # from positions keeps every metric consistent with the same source.
    dt = np.diff(t)
    dt = np.where(dt > 1e-4, dt, 1e-4)
    vx = np.diff(hx) / dt
    vy = np.diff(hy) / dt
    vz = np.diff(hz) / dt
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    speed = np.concatenate([[0.0], speed])

    if speed.max() <= 1e-6:
        return {}

    peak = speed.max()
    onset_thr = 0.10 * peak

    above = speed > onset_thr
    if not above.any():
        return {}
    onset_idx  = int(np.argmax(above))
    last_above = int(len(above) - 1 - np.argmax(above[::-1]))
    move_t = float(t[last_above] - t[onset_idx])

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

    seg = speed[onset_idx:last_above + 1]
    tpv_idx = onset_idx + int(np.argmax(seg))
    tpv_norm = (t[tpv_idx] - t[onset_idx]) / max(move_t, 1e-3)

    cross = trial[["cross_x", "cross_y", "cross_z"]].iloc[-1].to_numpy(dtype=float)
    final = np.array([hx[-1], hy[-1], hz[-1]])
    endpoint_dev = float(np.linalg.norm(final - cross))

    # Hit if the real hand passed within detection radius at any point.
    dists_to_cross = np.linalg.norm(
        np.column_stack([hx, hy, hz]) - cross[None, :], axis=1)
    hit = bool((dists_to_cross < DETECTION_RADIUS_M).any())

    dpos = np.diff(np.column_stack([hx, hy, hz]), axis=0)
    path_length = float(np.sum(np.linalg.norm(dpos, axis=1)))
    direct = float(np.linalg.norm(np.array([hx[-1] - hx[onset_idx],
                                            hy[-1] - hy[onset_idx],
                                            hz[-1] - hz[onset_idx]])))
    straightness = direct / max(path_length, 1e-3)

    # Forearm compensation at trial end, in the horizontal plane.
    # `forearm_compensation_abs_deg` is the unsigned angular deviation
    # between the real forearm vector (elbow → real hand) and the ideal
    # pointing-at-cross direction (elbow → cross). It grows with both the
    # perturbation magnitude and the participant's effort to fight it.
    # `forearm_compensation_aligned_deg` is signed so positive values mean
    # the user veered in the direction that would cancel the displayed
    # mismatch (i.e. genuine perceptual compensation), negative means they
    # drifted *with* the perturbation. Computed only when the elbow column
    # set is present (newer logs).
    forearm_compensation_abs_deg     = float("nan")
    forearm_compensation_aligned_deg = float("nan")
    if {"elbow_real_x", "elbow_real_y", "elbow_real_z"}.issubset(trial.columns):
        ex = float(trial["elbow_real_x"].iloc[-1])
        ez = float(trial["elbow_real_z"].iloc[-1])
        fx_real = hx[-1] - ex
        fz_real = hz[-1] - ez
        fx_ideal = cross[0] - ex
        fz_ideal = cross[2] - ez
        n_real  = (fx_real  ** 2 + fz_real  ** 2) ** 0.5
        n_ideal = (fx_ideal ** 2 + fz_ideal ** 2) ** 0.5
        if n_real > 1e-4 and n_ideal > 1e-4:
            cos = max(-1.0, min(1.0,
                       (fx_real * fx_ideal + fz_real * fz_ideal) /
                       (n_real * n_ideal)))
            forearm_compensation_abs_deg = float(np.degrees(np.arccos(cos)))
            # Signed angle around world Y: positive = real veered CCW from
            # the ideal direction (viewed from above).
            cross_y = fx_ideal * fz_real - fz_ideal * fx_real
            signed_dev = forearm_compensation_abs_deg * (1 if cross_y >= 0 else -1)
            # The displayed hand is rotated by +theta CCW around the elbow.
            # To pull it back onto the cross, the user must rotate the real
            # forearm by -theta. So compensation aligned with the
            # perturbation direction is `-sign(theta) * signed_dev`.
            theta_sign = 1 if float(trial["probe_angle"].iloc[0]) >= 0 else -1
            forearm_compensation_aligned_deg = -theta_sign * signed_dev

    return dict(
        movement_time=move_t,
        num_submovements=int(n_sub),
        peak_speed=float(peak),
        time_to_peak_norm=float(tpv_norm),
        endpoint_dev=endpoint_dev,
        path_length=path_length,
        straightness=straightness,
        forearm_compensation_abs_deg=forearm_compensation_abs_deg,
        forearm_compensation_aligned_deg=forearm_compensation_aligned_deg,
        hit=hit,
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
        sub = df[(df["phase"].astype(str) == "ProbeActive")
                 & (df["probe_level"] > 0)].copy()
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

def plot_panels(df: pd.DataFrame, save: Path, title_suffix: str = "") -> None:
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
        ax.boxplot(data, tick_labels=[f"{th:g}" for th in thetas])
        means = [np.mean(g) for g in data]
        ax.plot(range(1, len(thetas) + 1), means, "o-", c="C3", lw=1.5,
                label="mean")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("|θ|  (deg)")

    suptitle = "Behavioural metrics vs. perturbation magnitude"
    if title_suffix:
        suptitle += f"  —  {title_suffix}"
    fig.suptitle(suptitle, y=1.00)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=150)
    plt.close(fig)
    print(f"[info] figure saved to {save}")


# ──────────────────────────────────────────────────────────────────────
#  Aggregation: per-|θ| stats, hit rate, events, detection thresholds
# ──────────────────────────────────────────────────────────────────────

def per_theta_summary(df: pd.DataFrame, events: pd.DataFrame) -> list[dict]:
    """Return one dict per |θ| level, with full per-metric stats, hit rate
    and event count from the (optional) events file."""
    out = []
    ev_by_theta = (events.assign(abs_theta=events["probe_angle"].abs())
                          .groupby("abs_theta")
                          .size()
                          .to_dict()) if not events.empty else {}

    for theta, group in df.groupby("abs_theta"):
        entry = {
            "abs_theta_deg": float(theta),
            "n_trials": int(len(group)),
            "hit_rate": float(group["hit"].mean()) if "hit" in group else None,
            "event_count": int(ev_by_theta.get(float(theta), 0)),
            "metrics": {},
        }
        for m in METRIC_NAMES:
            if m not in group:
                continue
            vals = group[m].to_numpy(dtype=float)
            entry["metrics"][m] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
            }
        out.append(entry)
    return out


# Expected sign of perception-driven change for each metric. Used to score
# permutation tests one-sidedly: deviations against the expected direction
# count as noise, not as a perceptual response.
_EXPECTED_SIGN = {
    "movement_time":     +1,   # slows down when correcting
    "num_submovements":  +1,   # extra corrective velocity peaks
    "peak_speed":        -1,   # cautious slower reach
    "time_to_peak_norm": +1,   # peak shifts later in the reach
    "endpoint_dev":      +1,   # real hand veers off the cross
    "path_length":       +1,   # curved trajectory is longer
    "straightness":      -1,   # less straight
    "forearm_compensation_abs_deg":     +1,  # bigger deviation when reacting
    "forearm_compensation_aligned_deg": +1,  # positive = correct-direction compensation
}


def _permutation_p(baseline: np.ndarray, test: np.ndarray,
                    expected_sign: int, n_perm: int = 2000,
                    rng: np.random.Generator | None = None) -> float:
    """One-sided permutation test on the difference of means.

    H0: baseline and test trials are exchangeable (same distribution).
    H1: mean(test) - mean(baseline) has the same sign as expected_sign and
        is at least as extreme as the observed value.

    For small samples (here n=5+5 → C(10,5)=252 unique splits) the exact
    distribution is enumerated; otherwise n_perm random permutations are
    drawn. Returns a p-value in [0, 1]; lower means stronger evidence
    against H0 in the predicted direction.
    """
    from itertools import combinations
    from math import comb

    if rng is None:
        rng = np.random.default_rng(0)
    b = np.asarray(baseline, dtype=float)
    t = np.asarray(test,     dtype=float)
    nb, nt = len(b), len(t)
    if nb < 2 or nt < 2:
        return 1.0

    obs = float(np.mean(t) - np.mean(b))
    if expected_sign != 0 and obs * expected_sign <= 0:
        return 1.0   # observed effect has the wrong sign — not detection

    pool = np.concatenate([b, t])
    n = nb + nt

    if comb(n, nb) <= 2000:
        idxs = np.arange(n)
        count, total = 0, 0
        for sel in combinations(idxs, nb):
            mask = np.zeros(n, dtype=bool)
            mask[list(sel)] = True
            d = float(np.mean(pool[~mask]) - np.mean(pool[mask]))
            if expected_sign == 0:
                count += int(abs(d) >= abs(obs))
            else:
                count += int(d * expected_sign >= obs * expected_sign)
            total += 1
        return count / total

    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        d = float(np.mean(pool[idx[nb:]]) - np.mean(pool[idx[:nb]]))
        if expected_sign == 0:
            count += int(abs(d) >= abs(obs))
        else:
            count += int(d * expected_sign >= obs * expected_sign)
    return (count + 1) / (n_perm + 1)


def estimate_detection_thresholds(metrics_df: pd.DataFrame,
                                   alpha: float = 0.05,
                                   n_perm: int = 2000) -> dict:
    """For each metric, run a one-sided permutation test of |θ|=0 (baseline)
    against every |θ|>0. The detection threshold is the smallest |θ| at
    which the test rejects H0 (p < alpha) in the perception-driven
    direction. Returns a dict with the thresholds, the full per-(metric,
    |θ|) p-value matrix, and method metadata."""
    thetas = sorted(metrics_df["abs_theta"].unique())
    if 0.0 not in thetas:
        return {
            "method":         "permutation_test_one_sided",
            "alpha":          alpha,
            "n_permutations": n_perm,
            "thresholds":     {m: None for m in METRIC_NAMES},
            "pvalues":        {m: {} for m in METRIC_NAMES},
        }

    rng       = np.random.default_rng(42)
    base_mask = metrics_df["abs_theta"] == 0.0
    pvalues:    dict[str, dict[str, float]] = {}
    thresholds: dict[str, float | None]     = {}

    for m in METRIC_NAMES:
        if m not in metrics_df.columns:
            thresholds[m] = None
            pvalues[m]    = {}
            continue
        baseline = metrics_df.loc[base_mask, m].to_numpy(dtype=float)
        sign     = _EXPECTED_SIGN.get(m, 0)
        thr      = None
        per_theta_p: dict[str, float] = {}
        for th in thetas:
            if th <= 0.0:
                continue
            test_vals = metrics_df.loc[metrics_df["abs_theta"] == th, m].to_numpy(dtype=float)
            p = _permutation_p(baseline, test_vals, sign, n_perm, rng)
            per_theta_p[f"{th:g}"] = round(float(p), 4)
            if thr is None and p < alpha:
                thr = float(th)
        thresholds[m] = thr
        pvalues[m]    = per_theta_p

    return {
        "method":         "permutation_test_one_sided",
        "alpha":          alpha,
        "n_permutations": n_perm,
        "thresholds":     thresholds,
        "pvalues":        pvalues,
    }


def events_summary(events: pd.DataFrame) -> list[dict]:
    if events.empty:
        return []
    rows = []
    for label, sub in events.groupby("label"):
        rows.append({
            "label":  str(label),
            "count":  int(len(sub)),
            "thetas": sorted({float(abs(a)) for a in sub["probe_angle"].tolist()}),
        })
    return sorted(rows, key=lambda r: -r["count"])


def build_results(df: pd.DataFrame, events: pd.DataFrame,
                   source_name: str) -> dict:
    per_theta = per_theta_summary(df, events)
    detection = estimate_detection_thresholds(df)
    return {
        "source_file":     source_name,
        "participant_label": _participant_label_from_filename(source_name),
        "n_probe_trials":  int(len(df)),
        "detection_radius_m": DETECTION_RADIUS_M,
        "per_theta":       per_theta,
        "events_summary":  events_summary(events),
        "detection":       detection,
        "interpretation_notes": _interpretation_notes(per_theta, detection),
    }


def _participant_label_from_filename(name: str) -> str:
    """Best-effort label from filename like 'experiment_josue_20260509.csv'."""
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "experiment":
        return parts[1]
    return stem


def _interpretation_notes(per_theta: list[dict], detection: dict) -> dict:
    """Compact qualitative summary that downstream consumers (or the memoir
    writer) can read at a glance."""
    notes: dict = {}
    thresholds = detection.get("thresholds", {})
    detected_at = [v for v in thresholds.values() if v is not None]
    if detected_at:
        notes["earliest_detection_deg"]  = float(min(detected_at))
        notes["metrics_significant"]     = sorted(
            [m for m, v in thresholds.items() if v is not None]
        )
    else:
        notes["earliest_detection_deg"] = None
        notes["metrics_significant"]    = []

    if per_theta:
        notes["hit_rate_per_theta"] = {
            f"{e['abs_theta_deg']:g}": (round(e["hit_rate"], 3)
                                         if e["hit_rate"] is not None else None)
            for e in per_theta
        }
    return notes


# ──────────────────────────────────────────────────────────────────────
#  Text rendering
# ──────────────────────────────────────────────────────────────────────

def render_text_report(results: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f" PROBE ANALYSIS — {results['participant_label'].upper()}")
    lines.append("=" * 72)
    lines.append(f"Source:           {results['source_file']}")
    lines.append(f"Probe trials:     {results['n_probe_trials']}")
    lines.append(f"Hit radius (m):   {results['detection_radius_m']}")
    lines.append("")
    lines.append("Per-|θ| summary (|θ| in degrees):")
    lines.append("")
    header = (f"  {'|θ|':>5}  {'n':>3}  {'hit%':>5}  {'events':>6}  "
              f"{'mt(s)':>9}  {'nsub':>9}  {'peak(m/s)':>11}  "
              f"{'tpv':>7}  {'ep(m)':>9}  {'path(m)':>9}  {'straight':>9}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for entry in results["per_theta"]:
        m = entry["metrics"]
        def cell(name, fmt):
            if name not in m:
                return "    -    "
            return fmt.format(m[name]["mean"], m[name]["std"])
        hit_pct = f"{entry['hit_rate']*100:>4.0f}%" if entry["hit_rate"] is not None else "  -  "
        lines.append(
            f"  {entry['abs_theta_deg']:>5.0f}  {entry['n_trials']:>3}  {hit_pct:>5}  "
            f"{entry['event_count']:>6}  "
            f"{cell('movement_time',     '{:>5.2f}±{:>3.2f}')}  "
            f"{cell('num_submovements',  '{:>5.1f}±{:>3.1f}')}  "
            f"{cell('peak_speed',        '{:>6.2f}±{:>4.2f}')}  "
            f"{cell('time_to_peak_norm', '{:>3.2f}±{:>3.2f}')}  "
            f"{cell('endpoint_dev',      '{:>5.3f}±{:>3.3f}')}  "
            f"{cell('path_length',       '{:>5.3f}±{:>3.3f}')}  "
            f"{cell('straightness',      '{:>5.3f}±{:>3.3f}')}"
        )

    lines.append("")
    detection = results.get("detection", {})
    method    = detection.get("method", "?")
    alpha     = detection.get("alpha", 0.05)
    lines.append(f"Detection thresholds — {method}, α = {alpha}")
    lines.append("(smallest |θ| where the metric differs significantly from")
    lines.append(" baseline in the perception-driven direction)")
    thresholds = detection.get("thresholds", {})
    pvalues    = detection.get("pvalues", {})
    lines.append(f"  {'metric':<20s}  {'threshold':>10s}  {'p at thr':>10s}")
    for m in METRIC_NAMES:
        thr = thresholds.get(m)
        if thr is None:
            lines.append(f"  {m:<20s}  {'none':>10s}  {'-':>10s}")
        else:
            p = pvalues.get(m, {}).get(f"{thr:g}", float('nan'))
            lines.append(f"  {m:<20s}  {('%.0f°' % thr):>10s}  {p:>10.4f}")

    notes = results.get("interpretation_notes", {})
    earliest = notes.get("earliest_detection_deg")
    if earliest is not None:
        lines.append("")
        lines.append(f"Earliest detected metric kicks in at |θ| = {earliest:g}°")
        lines.append(f"Metrics significant: {', '.join(notes['metrics_significant'])}")
    else:
        lines.append("")
        lines.append("No metric reached significance within the tested range.")

    if results["events_summary"]:
        lines.append("")
        lines.append("Side-channel annotations:")
        for ev in results["events_summary"]:
            thetas = ", ".join(f"{th:g}°" for th in ev["thetas"])
            lines.append(f"  {ev['label']:<28s} count={ev['count']:>2}   thetas: {thetas}")
    else:
        lines.append("")
        lines.append("Side-channel annotations: none recorded.")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def _process_single(csv: Path, events_paths: list[Path],
                    fig_dir: Path, res_dir: Path,
                    fig_override: Path | None) -> None:
    df = load_probe_metrics([csv])
    print(f"[info] {csv.name}: {len(df)} probe trials parsed.")

    matching_events = [p for p in events_paths
                       if p.stem.split("_", 1)[-1] == csv.stem.split("_", 1)[-1]]
    events = load_events(matching_events) if matching_events else pd.DataFrame()

    results = build_results(df, events, csv.name)
    label = results["participant_label"]

    fig_path = fig_override if fig_override is not None else fig_dir / f"{csv.stem}.png"
    plot_panels(df, fig_path, title_suffix=label)

    res_dir.mkdir(parents=True, exist_ok=True)
    json_path = res_dir / f"{csv.stem}.json"
    txt_path  = res_dir / f"{csv.stem}.txt"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    txt_path.write_text(render_text_report(results), encoding="utf-8")
    print(f"[info] results written to {json_path}")
    print(f"[info] results written to {txt_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+", type=Path, help="experiment_*.csv files")
    ap.add_argument("--events", nargs="*", type=Path, default=[],
                    help="Optional events_*.csv files for spontaneous annotations.")
    ap.add_argument("--save", type=Path, default=None,
                    help="Override the figure output path (only valid with a single CSV).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR}).")
    args = ap.parse_args()

    fig_dir = args.out_dir / "grafics"
    res_dir = args.out_dir / "resultats"

    if args.save is not None and len(args.csvs) > 1:
        raise SystemExit("--save can only be used with a single input CSV.")

    for csv in args.csvs:
        _process_single(csv, list(args.events), fig_dir, res_dir,
                        args.save if len(args.csvs) == 1 else None)


if __name__ == "__main__":
    main()
