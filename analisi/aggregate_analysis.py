"""
Agregacio de resultats entre participants.

Llegeix els JSON i CSV per-trial generats per probe_analysis.py i:
  * Combina els quatre llindars (verbal, comportamental, cinematic, rest)
    en una taula resum per participant.
  * Creua les dades amb els atributs demografics (exp/participants_*.csv).
  * Genera figures agregades.
  * Executa tests no-parametrics per subgrups.
"""

from __future__ import annotations

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
PER_PART_DIR = PROJECT_ROOT / "data_analysis" / "per_participant"
OUT_DIR = PROJECT_ROOT / "data_analysis" / "aggregate"
PROBE_ANGLES = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0]

METRIC_TITLES = {
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


def perm_test_two_sided(a, b) -> float:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 1 or n_b < 1:
        return float("nan")
    pool = np.concatenate([a, b])
    n = n_a + n_b
    if n > 14:
        rng = np.random.default_rng(42)
        observed = abs(b.mean() - a.mean())
        count = 0; n_iter = 5000
        for _ in range(n_iter):
            perm = rng.permutation(pool)
            stat = abs(perm[n_a:].mean() - perm[:n_a].mean())
            if stat >= observed - 1e-12:
                count += 1
        return count / n_iter
    observed = abs(b.mean() - a.mean())
    count = 0; total = 0
    for combo in itertools.combinations(range(n), n_a):
        idx_a = set(combo)
        a_p = np.array([pool[i] for i in range(n) if i in idx_a])
        b_p = np.array([pool[i] for i in range(n) if i not in idx_a])
        stat = abs(b_p.mean() - a_p.mean())
        if stat >= observed - 1e-12:
            count += 1
        total += 1
    return count / total


def load_data():
    summary_files = sorted(glob.glob(str(PER_PART_DIR / "P*_summary.json")))
    summaries = []
    for f in summary_files:
        with open(f, encoding="utf-8") as fh:
            summaries.append(json.load(fh))
    kin_files = sorted(glob.glob(str(PER_PART_DIR / "P*_kinematic_per_trial.csv")))
    kin_dfs = [pd.read_csv(f) for f in kin_files]
    kin_all = pd.concat(kin_dfs, ignore_index=True) if kin_dfs else pd.DataFrame()
    rest_files = sorted(glob.glob(str(PER_PART_DIR / "P*_rest_per_trial.csv")))
    rest_dfs = [pd.read_csv(f) for f in rest_files]
    rest_all = pd.concat(rest_dfs, ignore_index=True) if rest_dfs else pd.DataFrame()
    demog_files = sorted(glob.glob(str(PROJECT_ROOT / "exp" / "participants_*.csv")))
    demog = pd.read_csv(demog_files[-1]) if demog_files else pd.DataFrame()
    return summaries, kin_all, rest_all, demog


def thresholds_table(summaries, demog) -> pd.DataFrame:
    rows = []
    for s in summaries:
        t = s["thresholds"]
        rows.append({
            "Codi": s["participant"],
            "thr_verbal": t.get("verbal"),
            "thr_behavioural": t.get("behavioural"),
            "thr_kinematic": t.get("kinematic_min"),
            "thr_rest": t.get("rest_min"),
            "n_trials": s.get("n_probe_trials"),
            "n_events": s.get("n_events"),
        })
    df = pd.DataFrame(rows)
    if not demog.empty:
        df = df.merge(demog, on="Codi", how="left")
    return df


def plot_thresholds_distribution(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for col, lab in [("thr_kinematic", "Cinematic"),
                     ("thr_behavioural", "Comportamental"),
                     ("thr_rest", "Repos"),
                     ("thr_verbal", "Verbal")]:
        vals = df[col].dropna().to_numpy()
        data.append(vals)
        labels.append(f"{lab}\n(n={len(vals)})")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    colors = ["#4FB477", "#D6453B", "#2E5EAA", "#F4A261"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for c in PROBE_ANGLES:
        ax.axhline(c, color="grey", lw=0.3, alpha=0.4, zorder=0)
    ax.set_ylabel("Llindar de deteccio (graus)")
    ax.set_title("Distribucio dels quatre tipus de llindar entre participants (n=10)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_thresholds_per_participant(df, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    pids = df["Codi"].tolist()
    x = np.arange(len(pids))
    w = 0.2
    cols = [("thr_kinematic", "Cinematic", "#4FB477"),
            ("thr_behavioural", "Comportamental", "#D6453B"),
            ("thr_rest", "Repos", "#2E5EAA"),
            ("thr_verbal", "Verbal", "#F4A261")]
    for i, (col, lab, c) in enumerate(cols):
        vals = df[col].fillna(-5).to_numpy()
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(pids, rotation=0)
    ax.set_ylabel("Llindar de deteccio (graus)")
    ax.set_title("Llindars per participant (barres negatives = no detectat)")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(kin_all, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Metriques cinematiques vs angle acumulat - una corba per participant + mitjana grup",
                 fontsize=14, fontweight="bold")
    metrics = list(METRIC_TITLES.keys())
    pids = sorted(kin_all["participant"].unique())
    cmap = plt.get_cmap("tab10")
    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        for j, pid in enumerate(pids):
            sub = kin_all[kin_all["participant"] == pid]
            agg = sub.groupby("level_target_deg")[metric].mean()
            ax.plot(agg.index, agg.values, "-o", color=cmap(j % 10),
                    alpha=0.5, lw=1.0, ms=3, label=pid if i == 0 else None)
        grp = kin_all.groupby("level_target_deg")[metric].agg(["mean", "std"])
        ax.plot(grp.index, grp["mean"], "-", color="black", lw=2.5,
                label="Mitjana grup" if i == 0 else None)
        ax.fill_between(grp.index, grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                        color="black", alpha=0.10)
        ax.set_xlabel("Angle acumulat (graus)")
        ax.set_ylabel(METRIC_TITLES[metric])
        ax.tick_params(labelsize=15)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=13, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_demog_correlations(thr_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Correlacio del llindar cinematic amb variables demografiques continues",
                 fontsize=12, fontweight="bold")
    var_pairs = [("Edat", "Edat (anys)"),
                 ("Alcada_cm", "Alcada (cm)"),
                 ("n_events", "Nombre d'anotacions registrades")]
    for ax, (col, label) in zip(axes, var_pairs):
        if col not in thr_df.columns:
            ax.set_title(f"(no disponible: {col})"); continue
        sub = thr_df.dropna(subset=[col, "thr_kinematic"])
        if len(sub) < 3:
            ax.set_title(f"(n insuf: {label})"); continue
        x = sub[col].astype(float).to_numpy()
        y = sub["thr_kinematic"].astype(float).to_numpy()
        ax.scatter(x, y, s=80, c="#2E5EAA", alpha=0.75, edgecolors="white")
        for pid, xi, yi in zip(sub["Codi"], x, y):
            ax.annotate(pid, (xi, yi), fontsize=11, xytext=(4, 4), textcoords="offset points")
        if len(x) >= 3:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.set_title(f"{label}  (r = {r:+.2f})")
        else:
            ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Llindar cinematic (graus)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_subgroup_comparison(thr_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Comparacio de llindars per subgrups demografics (n petit, exploratori)",
                 fontsize=12, fontweight="bold")
    groupings = [
        ("Genere", ["Masculi", "Femeni"]),
        ("Ma_dominant", ["Dreta", "Esquerra"]),
        ("Experiencia_VR", ["Cap", "Poca", "Moderada", "Habitual"]),
    ]
    for ax, (col, order) in zip(axes, groupings):
        if col not in thr_df.columns:
            ax.set_title(f"(no disponible: {col})"); continue
        data, labels = [], []
        for grp in order:
            sub = thr_df[thr_df[col] == grp]["thr_kinematic"].dropna()
            if len(sub) > 0:
                data.append(sub.to_numpy())
                labels.append(f"{grp}\n(n={len(sub)})")
        if not data:
            ax.set_title(f"(no dades: {col})"); continue
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#D6E2F5"); patch.set_edgecolor("#2E5EAA")
        ax.set_title(col)
        ax.set_ylabel("Llindar cinematic (graus)")
        ax.grid(True, alpha=0.3, axis="y")
        if len(data) == 2:
            p = perm_test_two_sided(data[0], data[1])
            ax.text(0.5, 0.95, f"perm test p = {p:.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=12, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_rest_metrics_curves(rest_all, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Metriques de la zona de repos (post-trial) vs angle acumulat",
                 fontsize=13, fontweight="bold")
    metrics = ["rest_path_length", "rest_settling_time", "rest_drift_total", "rest_final_offset"]
    titles = ["Long. recorregut al repos (m)", "Temps estabilitzacio (s)",
              "Drift de l'elbow (m)", "Offset final respecte centre (m)"]
    pids = sorted(rest_all["participant"].unique())
    cmap = plt.get_cmap("tab10")
    for ax, metric, title in zip(axes, metrics, titles):
        for j, pid in enumerate(pids):
            sub = rest_all[rest_all["participant"] == pid]
            agg = sub.groupby("level_target_deg")[metric].mean()
            ax.plot(agg.index, agg.values, "-o", color=cmap(j % 10), alpha=0.45, lw=1.0, ms=3)
        grp = rest_all.groupby("level_target_deg")[metric].agg(["mean", "std"])
        ax.plot(grp.index, grp["mean"], "-", color="black", lw=2.5)
        ax.fill_between(grp.index, grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                        color="black", alpha=0.10)
        ax.set_xlabel("Angle acumulat (graus)")
        ax.set_ylabel(title)
        ax.tick_params(labelsize=15)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, kin_all, rest_all, demog = load_data()
    if not summaries:
        print("[ERROR] No s'han trobat resultats per-participant.")
        return

    thr_df = thresholds_table(summaries, demog)
    thr_df.to_csv(OUT_DIR / "thresholds_table.csv", index=False)

    print("\n=== RESUM DE LLINDARS ===")
    for col, lab in [("thr_kinematic", "Cinematic"),
                     ("thr_behavioural", "Comportamental"),
                     ("thr_rest", "Repos"),
                     ("thr_verbal", "Verbal")]:
        vals = thr_df[col].dropna()
        if len(vals) > 0:
            print(f"  {lab:>15}: n_detectat={len(vals):2d}  mitjana={vals.mean():.1f}  "
                  f"mediana={vals.median():.1f}  min={vals.min():.0f}  max={vals.max():.0f}")

    plot_thresholds_distribution(thr_df, OUT_DIR / "thresholds_distribution.png")
    plot_thresholds_per_participant(thr_df, OUT_DIR / "thresholds_per_participant.png")
    if not kin_all.empty:
        plot_metric_curves(kin_all, OUT_DIR / "metrics_vs_angle.png")
    if not rest_all.empty:
        plot_rest_metrics_curves(rest_all, OUT_DIR / "rest_metrics_vs_angle.png")
    plot_demog_correlations(thr_df, OUT_DIR / "demographic_correlations.png")
    plot_subgroup_comparison(thr_df, OUT_DIR / "subgroup_comparison.png")

    aggregate_json = {
        "n_participants": len(summaries),
        "threshold_stats": {
            col.replace("thr_", ""): {
                "n_detected": int(thr_df[col].count()),
                "mean": float(thr_df[col].mean()) if thr_df[col].count() > 0 else None,
                "median": float(thr_df[col].median()) if thr_df[col].count() > 0 else None,
                "std": float(thr_df[col].std()) if thr_df[col].count() > 0 else None,
                "min": float(thr_df[col].min()) if thr_df[col].count() > 0 else None,
                "max": float(thr_df[col].max()) if thr_df[col].count() > 0 else None,
            }
            for col in ["thr_kinematic", "thr_behavioural", "thr_rest", "thr_verbal"]
        },
    }
    # Demographic breakdowns
    if "Genere" in thr_df.columns:
        aggregate_json["by_gender"] = thr_df.groupby("Genere")["thr_kinematic"].agg(["count","mean","std"]).reset_index().to_dict(orient="records")
    if "Ma_dominant" in thr_df.columns:
        aggregate_json["by_handedness"] = thr_df.groupby("Ma_dominant")["thr_kinematic"].agg(["count","mean","std"]).reset_index().to_dict(orient="records")
    if "Experiencia_VR" in thr_df.columns:
        aggregate_json["by_vr_exp"] = thr_df.groupby("Experiencia_VR")["thr_kinematic"].agg(["count","mean","std"]).reset_index().to_dict(orient="records")

    with open(OUT_DIR / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_json, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Aggregate fet. Sortida: {OUT_DIR}")


if __name__ == "__main__":
    main()
