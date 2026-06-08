# Simulation of Human Arm Movement in a Virtual Environment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![Unity](https://img.shields.io/badge/Unity-6-green?style=flat-square&logo=unity&logoColor=white)
![Motive](https://img.shields.io/badge/OptiTrack-Motive%20%2F%20NatNet-red?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bachelor's Thesis (TFG) — Bachelor's Degree in Computer Engineering, Universitat de Barcelona.**
</div>

A Virtual Reality platform built in **Unity 6** and integrated with an **OptiTrack** 6-camera optical tracking system, designed to study motor perception and the sense of body ownership. The system renders a first-person virtual arm driven by three physical markers (shoulder, elbow, hand) and, in a second phase, applies a controlled **real-time visuomotor perturbation** that dissociates the user's actual movement from the arm they see in the headset.

The central question is *how far the visual feedback of the arm can be rotated before the motor system reacts to it, and how much further before the participant becomes consciously aware of the mismatch.*

---

## How it works

```
OptiTrack cameras ──IR──> Motive ──NatNet──> Python client ──UDP──> Unity 6 ──render──> VR headset
```

1. Six OptiTrack cameras track three retro-reflective markers on the arm; **Motive** reconstructs their 3D positions.
2. A **Python NatNet client** filters the markers (including the headset's stray IR reflections) and forwards them to Unity over UDP.
3. A **point-correspondence calibration** (four table corners) aligns the OptiTrack and Unity coordinate systems, correcting for their opposite chirality.
4. The markers drive the rig of a humanoid avatar (Y Bot): joint identification, inverse kinematics for the elbow, and geometric roll stabilisation produce a believable first-person arm.
5. **Perturbation phase:** each frame, the displayed forearm is rotated around the *live elbow pivot* by a cumulative, monotonically growing angle θ, with a ramp synchronised to the spatial progress of the reach so the change stays sub-perceptual.

The perturbation is fully **analytical and real-time** — it requires no pre-trained model, no training data and no external process.

---

## Experimental paradigm

- **Baseline phase:** the participant rests the elbow in a green zone and reaches to touch a red cross that appears on the virtual table, with sound and silent trials.
- **Probe phase:** the same task, but the displayed hand is progressively rotated (11 levels, 5 trials each, +1.6° per trial, up to 80° accumulated).
- Every frame is logged to CSV (real and displayed hand, elbow, shoulder, headset pose, trial/phase). A side channel lets the experimenter annotate spontaneous reactions (hesitation, surprise, verbalisation).

The analysis estimates four **detection thresholds** per participant — kinematic, behavioural, rest-correction and verbal — using an exact one-sided permutation test.

> **Note:** the sessions reported here are **technical pilot trials** to validate the platform and paradigm, not a formal study with human subjects (no approved ethics protocol). Participant data is anonymised as `P00`–`P09`.

---

## Key result

Across a pilot cohort of ten participants, the **motor system reacts to the perturbation at ~17.6° of accumulated rotation on average, while conscious verbalisation of the mismatch only appears at ~53.6°** — a gap of roughly 36°, consistent with the literature on implicit visuomotor adaptation.

---

## Repository structure

```
.
├── unityproject/              Unity 6 project (graphics engine + experiment logic)
│   ├── Assets/
│   │   ├── Scripts/           C# scripts (see below)
│   │   ├── Scenes/            VR scene (1:1 lab table, XR Origin)
│   │   ├── Settings/          URP and XR configuration
│   │   └── CompositionLayers/ Y Bot rigged mesh (Mixamo)
│   ├── Packages/              Unity package manifest
│   └── ProjectSettings/       Unity project settings
├── PythonClient/              OptiTrack NatNet client + UDP forwarder (+ mouse simulator)
├── analisi/                   Post-hoc analysis pipeline (pandas / numpy / matplotlib)
│   ├── probe_analysis.py      Per-session metrics, thresholds and figures
│   ├── aggregate_analysis.py  Cross-participant aggregation and group figures
│   └── trajectory_examples.py Example trajectory / velocity-profile figures
├── data_analysis/             Generated analysis outputs (per-participant + aggregate PNG/JSON/CSV)
├── exp/example_data/          One full anonymised session (all file formats) for reproduction
├── Thesis/                    CAT_Thesis.pdf and CAT_Manual_Participant.pdf
├── LICENSE
└── README.md
```

### Unity C# scripts (`unityproject/Assets/Scripts/`)
| Script | Responsibility |
|---|---|
| `UDPMarkerReceiver.cs` | Threaded UDP listener; parses marker datagrams and filters headset ghost markers |
| `CoordinateSynchronizer.cs` | Point-correspondence calibration; maintains the rigid transform OptiTrack → Unity |
| `ArmModelController.cs` | Drives the Y Bot rig from the markers (joint ID, elbow IK, roll stabilisation, marker-loss handling) |
| `ArmPlaybackController.cs` | Applies the real-time visuomotor perturbation (forearm rotation about the live elbow) |
| `ExperimentController.cs` | Experiment state machine, cross spawning, scoring, CSV logging and annotation keys |
| `XRPositionLogger.cs` | Logs the headset pose |

---

## Quick start

### Requirements
- Unity 6 with the XR Interaction Toolkit
- A VR headset compatible with OpenXR (developed with an Oculus headset)
- OptiTrack with Motive and NatNet streaming enabled (for real tracking)
- Python 3.10+ with `pandas`, `numpy`, `matplotlib` (see `analisi/requirements.txt`)

### Running the tracking pipeline
1. On the OptiTrack PC, run `python PythonClient/PythonSample.py`.
2. Open `unityproject/` in Unity and press **Play**.
3. Place 4 markers on the table corners and press **C** to calibrate.
4. Place 3 markers on the arm (shoulder, elbow, hand) and press **O** to activate the virtual arm.
5. Press **B** to start the experiment.

For development without OptiTrack hardware, use `python PythonClient/PythonSampleMouse.py` to simulate marker data with the mouse over the same UDP protocol.

### Reproducing the analysis
```bash
pip install -r analisi/requirements.txt
python analisi/probe_analysis.py --all          # per-session metrics and figures
python analisi/aggregate_analysis.py            # cross-participant aggregation
```
The repository ships one anonymised session in `exp/example_data/` so the pipeline can be run end-to-end without access to the lab.

For the full description of the design, implementation and results, see the thesis: [`Thesis/CAT_Thesis.pdf`](Thesis/CAT_Thesis.pdf).

---

## Author

**Adrià Gasull Rectoret** · Bachelor's Thesis 2025–2026 · Universitat de Barcelona

Supervisor: Dr. Ignasi Cos Aguilera

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
