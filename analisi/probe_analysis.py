"""
Anàlisi de les dades experimentals d'un sol participant.

Llegeix experiment_*.csv + events_*.csv d'una sessió i:
  * Reconstrueix l'angle de pertorbació acumulada per trial (paradigma cumulatiu).
  * Calcula vuit mètriques cinemàtiques per trial probe.
  * Calcula quatre mètriques de correcció a la zona de repòs post-trial.
  * Estima quatre tipus de llindar de detecció:
      A. Llindar verbal       — primer angle amb 'user_said_strange'
      B. Llindar comportamental — primer angle amb qualsevol anotació
      C. Llindar cinemàtic    — primer angle amb mètrica significativa
                                (test de permutació exacte vs nivell control)
      D. Llindar de rest      — primer angle amb mètrica de rest significativa
  * Genera figures PNG per participant.
  * Desa JSON estructurat amb totes les mètriques.

Ús:
    python probe_analysis.py path/to/exp/P00_xxx/  [path/to/exp/P01_xxx/ ...]
    python probe_analysis.py --all                  # processa tota la carpeta exp/

El script és autocontingut: no depèn de scipy.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mida de text augmentada per a la llegibilitat de les figures a la memoria.
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 19,
})

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data_analysis"

# Configuració del paradigma (ha de coincidir amb el build de l'experiment).
PROBE_ANGLES = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0]
TRIALS_PER_LEVEL = 5
ALPHA = 0.05

# Per cada mètrica, direcció esperada de l'efecte amb angle creixent.
# LDLJ (Log Dimensionless Jerk) és NEGATIU: més negatiu = menys suau, per
# tant si la pertorbació redueix la suavitat l'esperem decreixent (-1).
METRIC_DIRECTIONS = {
    "movement_time":               +1,
    "num_submovements":            +1,
    "peak_speed":                  -1,
    "time_to_peak_norm":           +1,
    "endpoint_dev":                +1,
    "path_length":                 +1,
    "straightness":                -1,
    "forearm_compensation_deg":    +1,
    "ldlj_smoothness":             -1,
}

REST_METRIC_DIRECTIONS = {
    "rest_path_length":            +1,
    "rest_settling_time":          +1,
    "rest_drift_total":            +1,
    "rest_final_offset":           +1,
}


def held_angle_at_trial_end(level: int, trial_in_level: int) -> float:
    """Angle (graus) aplicat al final del trial K del nivell L."""
    if level < 1 or level > len(PROBE_ANGLES):
        return float("nan")
    target = PROBE_ANGLES[level - 1]
    prev = PROBE_ANGLES[level - 2] if level >= 2 else 0.0
    step = (target - prev) / TRIALS_PER_LEVEL
    return prev + step * trial_in_level


def trial_kinematics(trial: pd.DataFrame, cross_pos: np.ndarray) -> dict:
    if len(trial) < 5:
        return {}
    t = trial["timestamp"].to_numpy()
    hx = trial["hand_real_x"].to_numpy()
    hy = trial["hand_real_y"].to_numpy()
    hz = trial["hand_real_z"].to_numpy()
    ex = trial["elbow_real_x"].to_numpy()
    ez = trial["elbow_real_z"].to_numpy()
    dt = np.diff(t)
    dt[dt <= 0] = 1e-3
    vx = np.diff(hx) / dt
    vy = np.diff(hy) / dt
    vz = np.diff(hz) / dt
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    if speed.size < 3:
        return {}
    peak = float(speed.max())
    if peak < 1e-4:
        return {}
    thresh = 0.10 * peak
    above = speed >= thresh
    if not above.any():
        return {}
    onset_i = int(np.argmax(above))
    offset_i = int(len(above) - 1 - np.argmax(above[::-1]))
    if offset_i <= onset_i:
        return {}
    move_t = float(t[offset_i + 1] - t[onset_i])
    if move_t < 0.05:
        return {}
    thr_hi = 0.20 * peak
    thr_lo = 0.50 * peak
    in_peak = False
    peaks = 0
    for s in speed[onset_i:offset_i + 1]:
        if not in_peak and s > thr_hi:
            in_peak = True
            peaks += 1
        elif in_peak and s < thr_lo:
            in_peak = False
    peak_i = int(np.argmax(speed[onset_i:offset_i + 1]))
    ttp_norm = peak_i / max(1, (offset_i - onset_i))
    final_pos = np.array([hx[-1], hy[-1], hz[-1]])
    endpoint_dev = float(np.linalg.norm(final_pos - cross_pos))
    seg_lens = np.sqrt(np.diff(hx)**2 + np.diff(hy)**2 + np.diff(hz)**2)
    path_len = float(seg_lens.sum())
    straight_dist = float(np.linalg.norm(final_pos - np.array([hx[0], hy[0], hz[0]])))
    straightness = straight_dist / max(1e-6, path_len)
    elbow_xz = np.array([ex[-1], ez[-1]])
    hand_xz = np.array([hx[-1], hz[-1]])
    cross_xz = np.array([cross_pos[0], cross_pos[2]])
    v_hand = hand_xz - elbow_xz
    v_cross = cross_xz - elbow_xz
    n1 = np.linalg.norm(v_hand)
    n2 = np.linalg.norm(v_cross)
    if n1 < 1e-4 or n2 < 1e-4:
        compensation = 0.0
    else:
        cos_a = float(np.clip(np.dot(v_hand, v_cross) / (n1 * n2), -1, 1))
        compensation = float(np.degrees(np.arccos(cos_a)))

    # LDLJ (Log Dimensionless Jerk) — mesura de suavitat de la trajectoria.
    # Formulacio Hogan & Sternad (2009): LDLJ = -ln(T^5 / v_peak^2 * integral|jerk|^2 dt)
    # Es negatiu; mes negatiu => menys suau (mes esforc/correccions).
    ldlj = _compute_ldlj(
        hx[onset_i:offset_i + 2], hy[onset_i:offset_i + 2], hz[onset_i:offset_i + 2],
        t[onset_i:offset_i + 2], move_t, peak,
    )

    return dict(
        movement_time=move_t,
        num_submovements=peaks,
        peak_speed=peak,
        time_to_peak_norm=ttp_norm,
        endpoint_dev=endpoint_dev,
        path_length=path_len,
        straightness=straightness,
        forearm_compensation_deg=compensation,
        ldlj_smoothness=ldlj,
    )


def _compute_ldlj(hx, hy, hz, t, move_t, peak_speed) -> float:
    """Log Dimensionless Jerk sobre la finestra del reach (onset->offset).

    Calcula derivades successives via diferencies centrals/avancades, integra
    |jerk|^2 amb regla trapezoidal i normalitza segons Hogan & Sternad:
        eta = (T^5 / v_peak^2) * integral|jerk|^2 dt
        LDLJ = -ln(eta)
    Retorna NaN si la finestra es massa curta per a derivar tres cops.
    """
    n = len(hx)
    if n < 6 or move_t <= 0 or peak_speed <= 0:
        return float("nan")
    dt = np.diff(t)
    dt[dt <= 0] = 1e-3
    # velocitat (n-1 mostres)
    vx = np.diff(hx) / dt
    vy = np.diff(hy) / dt
    vz = np.diff(hz) / dt
    # acceleracio (n-2 mostres) — usem dt avancat
    dt2 = dt[1:]
    ax = np.diff(vx) / dt2
    ay = np.diff(vy) / dt2
    az = np.diff(vz) / dt2
    # jerk (n-3 mostres)
    dt3 = dt2[1:]
    jx = np.diff(ax) / dt3
    jy = np.diff(ay) / dt3
    jz = np.diff(az) / dt3
    if len(jx) < 1:
        return float("nan")
    j2 = jx**2 + jy**2 + jz**2
    # Integral trapezoidal de |jerk|^2 dt
    integral = float(np.trapezoid(j2, dx=float(np.mean(dt3)))) if hasattr(np, "trapezoid") \
               else float(np.trapz(j2, dx=float(np.mean(dt3))))
    if integral <= 0:
        return float("nan")
    eta = (move_t**5 / max(peak_speed**2, 1e-9)) * integral
    if eta <= 0:
        return float("nan")
    return float(-np.log(eta))


def rest_metrics(rest_block: pd.DataFrame) -> dict:
    if len(rest_block) < 5:
        return {}
    t = rest_block["timestamp"].to_numpy()
    hx = rest_block["hand_real_x"].to_numpy()
    hy = rest_block["hand_real_y"].to_numpy()
    hz = rest_block["hand_real_z"].to_numpy()
    ex = rest_block["elbow_real_x"].to_numpy()
    ez = rest_block["elbow_real_z"].to_numpy()
    dt = np.diff(t)
    dt[dt <= 0] = 1e-3
    speed_hand = np.sqrt(
        (np.diff(hx) / dt)**2 + (np.diff(hy) / dt)**2 + (np.diff(hz) / dt)**2
    )
    seg_lens = np.sqrt(np.diff(hx)**2 + np.diff(hy)**2 + np.diff(hz)**2)
    path_len = float(seg_lens.sum())
    settle_thresh = 0.02
    hold = 0.0
    settling_time = float(t[-1] - t[0])
    for i, s in enumerate(speed_hand):
        if s < settle_thresh:
            hold += float(dt[i])
            if hold >= 0.2:
                settling_time = float(t[i] - t[0])
                break
        else:
            hold = 0.0
    drift_total = float(np.sum(np.sqrt(np.diff(ex)**2 + np.diff(ez)**2)))
    rest_center_xz = np.array([0.15, 0.42])
    elbow_final_xz = np.array([ex[-1], ez[-1]])
    final_offset = float(np.linalg.norm(elbow_final_xz - rest_center_xz))
    return dict(
        rest_path_length=path_len,
        rest_settling_time=settling_time,
        rest_drift_total=drift_total,
        rest_final_offset=final_offset,
    )


def perm_test_one_sided(a, b, direction: int = +1) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float)
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n_a, n_b = len(a), len(b)
    pool = np.concatenate([a, b])
    n = len(pool)
    if n > 14:
        rng = np.random.default_rng(42)
        observed = direction * (b.mean() - a.mean())
        count = 0
        n_iter = 5000
        for _ in range(n_iter):
            perm = rng.permutation(pool)
            stat = direction * (perm[n_a:].mean() - perm[:n_a].mean())
            if stat >= observed - 1e-12:
                count += 1
        return count / n_iter
    observed = direction * (b.mean() - a.mean())
    count = 0
    total = 0
    for combo in itertools.combinations(range(n), n_a):
        idx_a = set(combo)
        a_p = np.array([pool[i] for i in range(n) if i in idx_a])
        b_p = np.array([pool[i] for i in range(n) if i not in idx_a])
        stat = direction * (b_p.mean() - a_p.mean())
        if stat >= observed - 1e-12:
            count += 1
        total += 1
    return count / total


def thresholds_kinematic(metrics_df: pd.DataFrame) -> dict:
    out = {}
    if "level" not in metrics_df.columns:
        return out
    baseline = metrics_df[metrics_df["level"] == 1]
    for metric, direction in METRIC_DIRECTIONS.items():
        threshold = None
        p_at_thr = None
        for lvl in sorted(metrics_df["level"].unique()):
            if lvl <= 1:
                continue
            test_vals = metrics_df.loc[metrics_df["level"] == lvl, metric].to_numpy(dtype=float)
            base_vals = baseline[metric].to_numpy(dtype=float)
            p = perm_test_one_sided(base_vals, test_vals, direction)
            if not np.isnan(p) and p < ALPHA:
                threshold = float(PROBE_ANGLES[lvl - 1])
                p_at_thr = float(p)
                break
        out[metric] = {"threshold_deg": threshold, "p_value": p_at_thr}
    return out


def threshold_min_kinematic(thr_kin: dict):
    vals = [v["threshold_deg"] for v in thr_kin.values() if v["threshold_deg"] is not None]
    return float(min(vals)) if vals else None


def threshold_verbal(events: pd.DataFrame):
    if events.empty or "label" not in events.columns:
        return None
    said = events[events["label"] == "user_said_strange"]
    if said.empty:
        return None
    return float(said["probe_angle"].min())


def threshold_behavioural(events: pd.DataFrame):
    if events.empty or "label" not in events.columns:
        return None
    rel = events[events["label"].isin(
        ["user_said_strange", "user_visibly_surprised", "user_hesitated"]
    )]
    if rel.empty:
        return None
    return float(rel["probe_angle"].min())


def analyse_participant(folder: Path, out_dir: Path) -> dict:
    pid = folder.name.split("_")[0]
    exp_files = list(folder.glob("experiment_*.csv"))
    if not exp_files:
        print(f"[{pid}] cap experiment_*.csv trobat — saltant.")
        return {}
    event_files = list(folder.glob("events_*.csv"))
    df = pd.read_csv(exp_files[0])
    events = pd.read_csv(event_files[0]) if event_files else pd.DataFrame()

    probe = df[df["phase"] == "ProbeActive"].copy()
    rows = []
    for (lvl, trl), grp in probe.groupby(["probe_level", "probe_trial"]):
        if trl > TRIALS_PER_LEVEL:
            continue
        cross = np.array([
            grp["cross_x"].iloc[0],
            grp["cross_y"].iloc[0],
            grp["cross_z"].iloc[0],
        ])
        m = trial_kinematics(grp, cross)
        if not m:
            continue
        m["participant"] = pid
        m["level"] = int(lvl)
        m["trial_in_level"] = int(trl)
        m["level_target_deg"] = float(PROBE_ANGLES[int(lvl) - 1])
        m["held_angle_deg"] = held_angle_at_trial_end(int(lvl), int(trl))
        rows.append(m)
    kin_df = pd.DataFrame(rows)

    rest_blocks = []
    current_rest = None
    last_level = None
    last_trial = None
    for _, row in df.iterrows():
        ph = row["phase"]
        if ph == "ProbeWaitRest":
            if current_rest is None:
                current_rest = []
            current_rest.append(row)
        else:
            if current_rest is not None and len(current_rest) > 0:
                rest_blocks.append((last_level, last_trial, pd.DataFrame(current_rest)))
                current_rest = None
        if ph == "ProbeActive":
            last_level = int(row["probe_level"])
            last_trial = int(row["probe_trial"])
    if current_rest is not None and len(current_rest) > 0:
        rest_blocks.append((last_level, last_trial, pd.DataFrame(current_rest)))

    rest_rows = []
    for lvl, trl, block in rest_blocks:
        if lvl is None or trl is None:
            continue
        m = rest_metrics(block)
        if not m:
            continue
        m["participant"] = pid
        m["level"] = int(lvl)
        m["trial_in_level"] = int(trl)
        m["level_target_deg"] = float(PROBE_ANGLES[int(lvl) - 1]) if 1 <= lvl <= len(PROBE_ANGLES) else float("nan")
        m["held_angle_deg"] = held_angle_at_trial_end(int(lvl), int(trl))
        rest_rows.append(m)
    rest_df = pd.DataFrame(rest_rows)

    thr_kin = thresholds_kinematic(kin_df) if not kin_df.empty else {}
    thr_kin_min = threshold_min_kinematic(thr_kin)

    thr_rest = {}
    if not rest_df.empty:
        baseline = rest_df[rest_df["level"] == 1]
        for metric, direction in REST_METRIC_DIRECTIONS.items():
            thr = None
            p_at = None
            for lvl in sorted(rest_df["level"].unique()):
                if lvl <= 1:
                    continue
                test_vals = rest_df.loc[rest_df["level"] == lvl, metric].to_numpy(dtype=float)
                base_vals = baseline[metric].to_numpy(dtype=float)
                p = perm_test_one_sided(base_vals, test_vals, direction)
                if not np.isnan(p) and p < ALPHA:
                    thr = float(PROBE_ANGLES[lvl - 1])
                    p_at = float(p)
                    break
            thr_rest[metric] = {"threshold_deg": thr, "p_value": p_at}
    thr_rest_min = None
    rvals = [v["threshold_deg"] for v in thr_rest.values() if v["threshold_deg"] is not None]
    if rvals:
        thr_rest_min = float(min(rvals))

    thr_verbal = threshold_verbal(events)
    thr_behav = threshold_behavioural(events)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{pid}_metrics.png"
    if not kin_df.empty:
        plot_participant(pid, kin_df, fig_path)

    summary = {
        "participant": pid,
        "n_probe_trials": int(len(kin_df)),
        "n_rest_periods": int(len(rest_df)),
        "n_events": int(len(events)),
        "event_labels": events["label"].value_counts().to_dict() if not events.empty else {},
        "thresholds": {
            "verbal": thr_verbal,
            "behavioural": thr_behav,
            "kinematic_min": thr_kin_min,
            "rest_min": thr_rest_min,
        },
        "thresholds_per_metric": thr_kin,
        "thresholds_per_rest_metric": thr_rest,
    }

    if not kin_df.empty:
        kin_df.to_csv(out_dir / f"{pid}_kinematic_per_trial.csv", index=False)
    if not rest_df.empty:
        rest_df.to_csv(out_dir / f"{pid}_rest_per_trial.csv", index=False)

    json_path = out_dir / f"{pid}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"[{pid}] {len(kin_df)} trials kin, {len(rest_df)} blocs rest, {len(events)} events  "
          f"-> thr_kin_min={thr_kin_min}, thr_rest_min={thr_rest_min}, "
          f"thr_verbal={thr_verbal}, thr_behav={thr_behav}")
    return summary


def plot_participant(pid: str, kin_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f"Participant {pid} — metriques cinematiques vs angle acumulat",
                 fontsize=14, fontweight="bold")
    metrics = list(METRIC_DIRECTIONS.keys())
    titles = {
        "movement_time": "Temps moviment (s)",
        "num_submovements": "Sub-moviments",
        "peak_speed": "Velocitat pic (m/s)",
        "time_to_peak_norm": "TPV normalitzat",
        "endpoint_dev": "Desviacio endpoint (m)",
        "path_length": "Long. recorregut (m)",
        "straightness": "Rectitud",
        "forearm_compensation_deg": "Compensacio avantbrac (graus)",
        "ldlj_smoothness": "LDLJ smoothness",
    }
    levels_sorted = sorted(kin_df["level"].unique())
    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        data = []
        labels = []
        for lvl in levels_sorted:
            sub = kin_df[kin_df["level"] == lvl][metric].dropna()
            data.append(sub.to_numpy())
            labels.append(f"{PROBE_ANGLES[lvl-1]:.0f}")
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, lvl in zip(bp["boxes"], levels_sorted):
            patch.set_facecolor("#D6E2F5" if lvl > 1 else "#DEF5E5")
            patch.set_edgecolor("#2E5EAA")
        ax.set_xlabel("Angle acumulat (graus)")
        ax.set_ylabel(titles[metric])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", help="Carpetes de participant (P00_xxx)")
    ap.add_argument("--all", action="store_true", help="Processa totes les carpetes a exp/")
    ap.add_argument("-o", "--out-dir", default=str(DEFAULT_OUTPUT / "per_participant"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.all:
        folders = sorted([Path(p) for p in glob.glob(str(PROJECT_ROOT / "exp" / "P*"))
                          if Path(p).is_dir()])
    else:
        folders = [Path(p) for p in args.folders]
    all_summaries = []
    for folder in folders:
        s = analyse_participant(folder, out_dir)
        if s:
            all_summaries.append(s)
    master = out_dir / "all_participants_summary.json"
    with open(master, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] {len(all_summaries)} participants processats. Master: {master}")


if __name__ == "__main__":
    main()
