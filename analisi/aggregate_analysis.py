"""
Cross-participant aggregation of probe-phase results.

Reads every JSON produced by `probe_analysis.py` (one per participant, in
`data_analysis/resultats/`) and produces:

    grafics/
        metrics_per_theta.png   — 4 metrics × all participants vs. |θ|
        detection_thresholds.png — per-participant detection threshold per metric
        hit_rate.png             — hit rate vs. |θ| with group mean overlay
        profile_correlations.png — scatter plots of profile attributes vs.
                                   the earliest detection level

    resultats/
        agregat.json             — group-level statistics and per-participant table
        agregat.txt              — human-readable digest of the same content

Usage:
    python aggregate_analysis.py
        Reads from <project_root>/data_analysis/resultats/
        Writes to <project_root>/data_analysis/agregat/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR  = PROJECT_ROOT / "data_analysis" / "resultats"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis" / "agregat"

PRIMARY_METRICS = [
    "movement_time", "num_submovements", "endpoint_dev", "straightness",
]
PRIMARY_TITLES = {
    "movement_time":    "Movement time (s)",
    "num_submovements": "Sub-movements per trial",
    "endpoint_dev":     "Endpoint deviation (m)",
    "straightness":     "Trajectory straightness",
}

# Participant attributes are loaded at runtime from `participants.json`
# living next to this script. Update that file with one entry per session
# you run; the cross-participant correlation panel uses these attributes
# to look for relationships with the kinematic detection thresholds.
# If the file is missing or empty, profile correlations are skipped.
PARTICIPANTS_FILE = Path(__file__).resolve().parent / "participants.json"


def load_participant_attrs() -> dict:
    """Return a dict keyed by participant label (e.g. firstname) with the
    attribute fields used downstream. Empty dict if the file does not
    exist or has no `participants` map."""
    if not PARTICIPANTS_FILE.exists():
        return {}
    try:
        data = json.loads(PARTICIPANTS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] could not parse {PARTICIPANTS_FILE.name}: {e}")
        return {}
    return data.get("participants", {}) or {}


PARTICIPANT_ATTRS: dict = load_participant_attrs()
# Scales used in the JSON file:
#   vr_level: 0 none, 1 occasional, 2 experienced, 3 expert.
#   anxiety:  1 low, 2 moderate, 3 high.


# ──────────────────────────────────────────────────────────────────────
#  Loading
# ──────────────────────────────────────────────────────────────────────

def load_all_results(in_dir: Path) -> list[dict]:
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON results found under {in_dir}.")
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[warn] could not read {f}: {e}")
    return out


def metric_matrix(results: list[dict], metric: str) -> tuple[list[float], list[str], np.ndarray]:
    """Return (thetas, labels, values[participants, thetas]) for one metric.
    Cells are NaN where a participant has no data at a given |θ|."""
    thetas = sorted({e["abs_theta_deg"]
                     for r in results for e in r["per_theta"]})
    labels = [r["participant_label"] for r in results]
    values = np.full((len(results), len(thetas)), np.nan)
    for i, r in enumerate(results):
        per = {e["abs_theta_deg"]: e for e in r["per_theta"]}
        for j, th in enumerate(thetas):
            if th in per and metric in per[th]["metrics"]:
                values[i, j] = per[th]["metrics"][metric]["mean"]
    return thetas, labels, values


def hit_rate_matrix(results: list[dict]) -> tuple[list[float], list[str], np.ndarray]:
    thetas = sorted({e["abs_theta_deg"]
                     for r in results for e in r["per_theta"]})
    labels = [r["participant_label"] for r in results]
    values = np.full((len(results), len(thetas)), np.nan)
    for i, r in enumerate(results):
        per = {e["abs_theta_deg"]: e for e in r["per_theta"]}
        for j, th in enumerate(thetas):
            if th in per and per[th]["hit_rate"] is not None:
                values[i, j] = per[th]["hit_rate"]
    return thetas, labels, values


# ──────────────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────────────

def plot_metrics_per_theta(results: list[dict], save: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, metric in zip(axes.flat, PRIMARY_METRICS):
        thetas, labels, vals = metric_matrix(results, metric)
        # Per-participant lines
        for i, lbl in enumerate(labels):
            ax.plot(thetas, vals[i, :], "o-", lw=1.0, alpha=0.55, label=lbl)
        # Group mean overlay
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(vals, axis=0)
        ax.plot(thetas, mean, "k-", lw=2.5, label="group mean")
        ax.set_title(PRIMARY_TITLES[metric])
        ax.set_xlabel("|θ|  (deg)")
        ax.set_ylabel(PRIMARY_TITLES[metric])
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle("Cross-participant kinematic response vs. perturbation magnitude",
                 y=1.00)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=150)
    plt.close(fig)
    print(f"[info] saved {save}")


def plot_detection_thresholds(results: list[dict], save: Path) -> None:
    metrics = list(PRIMARY_METRICS)
    labels  = [r["participant_label"] for r in results]
    n_p, n_m = len(labels), len(metrics)
    data = np.full((n_p, n_m), np.nan)
    for i, r in enumerate(results):
        thr = r.get("detection", {}).get("thresholds", {})
        for j, m in enumerate(metrics):
            v = thr.get(m)
            if v is not None:
                data[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    x = np.arange(n_p)
    for j, m in enumerate(metrics):
        bars = data[:, j]
        # Render NaN as 0-height with a hatched bar at the top of the axis.
        plotted = np.where(np.isnan(bars), 0, bars)
        ax.bar(x + (j - 1.5) * width, plotted, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detection threshold |θ|  (deg)   —  lower = more sensitive")
    ax.set_title("Per-participant detection thresholds per metric (permutation test, α=0.05)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=150)
    plt.close(fig)
    print(f"[info] saved {save}")


def plot_hit_rate(results: list[dict], save: Path) -> None:
    thetas, labels, vals = hit_rate_matrix(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, lbl in enumerate(labels):
        ax.plot(thetas, vals[i, :] * 100.0, "o-", lw=1.0, alpha=0.6, label=lbl)
    with np.errstate(invalid="ignore"):
        mean_pct = np.nanmean(vals, axis=0) * 100.0
    ax.plot(thetas, mean_pct, "k-", lw=2.5, label="group mean")
    ax.set_xlabel("|θ|  (deg)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Hit rate vs. perturbation magnitude")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=150)
    plt.close(fig)
    print(f"[info] saved {save}")


def plot_profile_correlations(results: list[dict], save: Path) -> None:
    """Scatter the earliest detection |θ| per participant against three
    profile attributes (age, VR experience, anxiety). Annotate each point
    with the participant label."""
    pts = []
    for r in results:
        lbl = r["participant_label"]
        attrs = PARTICIPANT_ATTRS.get(lbl)
        if attrs is None:
            continue
        earliest = r.get("interpretation_notes", {}).get("earliest_detection_deg")
        if earliest is None:
            continue
        pts.append((lbl, attrs, float(earliest)))
    if not pts:
        print("[warn] no profile-correlation points to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, attr, label in zip(
        axes,
        ["age", "vr_level", "anxiety"],
        ["Age (years)", "VR experience (0=none .. 3=expert)",
         "Anxiety estimate (1=low .. 3=high)"],
    ):
        xs = [a[attr] for _, a, _ in pts]
        ys = [det for _, _, det in pts]
        ax.scatter(xs, ys, s=70, c="C0", edgecolor="k", zorder=3)
        for (lbl, _, det), x in zip(pts, xs):
            ax.annotate(lbl, (x, det), xytext=(5, 5), textcoords="offset points",
                        fontsize=8)
        # Pearson correlation if more than 2 unique x values
        if len(set(xs)) >= 3:
            cor = float(np.corrcoef(xs, ys)[0, 1])
            ax.set_title(f"{label}\nPearson r = {cor:+.2f}")
        else:
            ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Earliest detection |θ| (deg)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Profile attributes vs. perceptual sensitivity", y=1.02)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] saved {save}")


# ──────────────────────────────────────────────────────────────────────
#  Group-level summary
# ──────────────────────────────────────────────────────────────────────

def group_stats(results: list[dict]) -> dict:
    """Group-level (mean ± SD across participants) per |θ| per metric."""
    thetas = sorted({e["abs_theta_deg"]
                     for r in results for e in r["per_theta"]})
    out = {"thetas": thetas, "n_participants": len(results), "metrics": {}}
    for metric in PRIMARY_METRICS + ["peak_speed", "time_to_peak_norm", "path_length"]:
        _, _, vals = metric_matrix(results, metric)
        per_theta = []
        for j in range(len(thetas)):
            col = vals[:, j]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                per_theta.append({"mean": None, "std": None, "n": 0})
            else:
                per_theta.append({
                    "mean": float(np.mean(col)),
                    "std":  float(np.std(col, ddof=1)) if len(col) > 1 else 0.0,
                    "n":    int(len(col)),
                })
        out["metrics"][metric] = per_theta

    # Hit rate
    _, _, hr = hit_rate_matrix(results)
    out["hit_rate"] = []
    for j in range(len(thetas)):
        col = hr[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            out["hit_rate"].append({"mean": None, "std": None, "n": 0})
        else:
            out["hit_rate"].append({
                "mean": float(np.mean(col)),
                "std":  float(np.std(col, ddof=1)) if len(col) > 1 else 0.0,
                "n":    int(len(col)),
            })
    return out


def per_participant_table(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        lbl = r["participant_label"]
        notes = r.get("interpretation_notes", {})
        det = r.get("detection", {})
        ev_total = sum(e["count"] for e in r.get("events_summary", []))
        out.append({
            "participant":            lbl,
            "attributes":             PARTICIPANT_ATTRS.get(lbl, {}),
            "n_probe_trials":         r.get("n_probe_trials"),
            "earliest_detection_deg": notes.get("earliest_detection_deg"),
            "metrics_significant":    notes.get("metrics_significant", []),
            "thresholds":             det.get("thresholds", {}),
            "pvalues":                det.get("pvalues", {}),
            "events_total":           ev_total,
            "hit_rate_per_theta":     notes.get("hit_rate_per_theta", {}),
        })
    return out


def correlate_profile_vs_detection(results: list[dict]) -> dict:
    pts = []
    for r in results:
        attrs = PARTICIPANT_ATTRS.get(r["participant_label"])
        det = r.get("interpretation_notes", {}).get("earliest_detection_deg")
        if attrs is not None and det is not None:
            pts.append((attrs, float(det)))
    if len(pts) < 3:
        return {}
    out = {}
    for attr in ["age", "vr_level", "anxiety", "height_m"]:
        xs = np.array([a[attr] for a, _ in pts], dtype=float)
        ys = np.array([d       for _, d in pts], dtype=float)
        if len(set(xs.tolist())) < 2:
            continue
        cor = float(np.corrcoef(xs, ys)[0, 1])
        out[attr] = round(cor, 3)
    return out


def build_aggregate(results: list[dict]) -> dict:
    return {
        "n_participants":     len(results),
        "participants":       [r["participant_label"] for r in results],
        "group_stats":        group_stats(results),
        "per_participant":    per_participant_table(results),
        "profile_vs_detection_pearson": correlate_profile_vs_detection(results),
    }


# ──────────────────────────────────────────────────────────────────────
#  Text rendering
# ──────────────────────────────────────────────────────────────────────

def render_text_report(agg: dict) -> str:
    L = []
    L.append("=" * 72)
    L.append(" CROSS-PARTICIPANT PROBE-PHASE AGGREGATE")
    L.append("=" * 72)
    L.append(f"Participants:  {agg['n_participants']}")
    L.append("Labels:        " + ", ".join(agg["participants"]))
    L.append("")
    L.append("Group statistics (mean ± SD across participants) per |θ|:")
    L.append("")
    g = agg["group_stats"]
    thetas = g["thetas"]
    header = f"  {'|θ|':>5}  " + "".join(f"{m:>22}" for m in PRIMARY_METRICS)
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for j, th in enumerate(thetas):
        cells = []
        for m in PRIMARY_METRICS:
            entry = g["metrics"][m][j]
            if entry["mean"] is None:
                cells.append(" " * 22)
            else:
                cells.append(f"   {entry['mean']:>8.3f} ± {entry['std']:>6.3f}")
        L.append(f"  {th:>5.0f}" + "".join(cells))

    L.append("")
    L.append("Group hit rate (mean ± SD across participants):")
    L.append(f"  {'|θ|':>5}  {'hit rate (%)':>14}")
    L.append("  " + "-" * 22)
    for j, th in enumerate(thetas):
        e = g["hit_rate"][j]
        if e["mean"] is None:
            L.append(f"  {th:>5.0f}  {'-':>14}")
        else:
            L.append(f"  {th:>5.0f}  "
                     f"{e['mean']*100:>6.1f} ± {e['std']*100:>5.1f}")

    L.append("")
    L.append("Per-participant detection thresholds")
    L.append("(one-sided permutation test, α = 0.05):")
    L.append(f"  {'name':>10}  {'age':>4}  {'VR':>3}  {'anx':>4}  "
             f"{'earliest':>9}  {'#metrics':>9}  {'#events':>8}")
    L.append("  " + "-" * 60)
    for p in agg["per_participant"]:
        attrs = p.get("attributes", {})
        earliest = (f"{p['earliest_detection_deg']:.0f}°"
                    if p.get("earliest_detection_deg") is not None else "  -")
        L.append(f"  {p['participant']:>10}  "
                 f"{attrs.get('age','-'):>4}  "
                 f"{attrs.get('vr_level','-'):>3}  "
                 f"{attrs.get('anxiety','-'):>4}  "
                 f"{earliest:>9}  "
                 f"{len(p.get('metrics_significant', [])):>9}  "
                 f"{p.get('events_total', 0):>8}")

    cors = agg.get("profile_vs_detection_pearson", {})
    if cors:
        L.append("")
        L.append("Pearson correlation between profile attributes and earliest")
        L.append("detection |θ| (negative = attribute → earlier detection):")
        for attr, r in cors.items():
            L.append(f"  {attr:<12s}  r = {r:+.2f}")

    L.append("")
    L.append("Notes:")
    L.append("  * 'earliest' = smallest |θ| where any metric exceeds")
    L.append("     baseline at p < 0.05 with the perception-driven sign;")
    L.append("     lower values mean earlier perceptual response.")
    L.append("  * '#metrics' = how many of the 7 tracked metrics crossed that")
    L.append("     threshold within the tested range (max 7).")
    L.append("  * Hit rate is computed on the REAL hand (not the displayed");
    L.append("     one), so it drops when the participant compensates the")
    L.append("     visual mismatch and the real hand veers off the cross.")
    L.append("")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_INPUT_DIR,
                    help=f"Folder with per-participant JSONs (default: {DEFAULT_INPUT_DIR}).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Folder for aggregated outputs (default: {DEFAULT_OUTPUT_DIR}).")
    args = ap.parse_args()

    results = load_all_results(args.in_dir)
    print(f"[info] loaded {len(results)} participant result files")

    fig_dir = args.out_dir / "grafics"
    res_dir = args.out_dir / "resultats"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics_per_theta   (results, fig_dir / "metrics_per_theta.png")
    plot_detection_thresholds(results, fig_dir / "detection_thresholds.png")
    plot_hit_rate            (results, fig_dir / "hit_rate.png")
    plot_profile_correlations(results, fig_dir / "profile_correlations.png")

    agg = build_aggregate(results)
    (res_dir / "agregat.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    (res_dir / "agregat.txt").write_text(
        render_text_report(agg), encoding="utf-8")
    print(f"[info] wrote {res_dir / 'agregat.json'}")
    print(f"[info] wrote {res_dir / 'agregat.txt'}")


if __name__ == "__main__":
    main()
