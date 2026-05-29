# To Adapt or Not: Execution-Level Verification for Test-Time Adaptation of Vision-Language-Action Models

[![Code](https://img.shields.io/badge/Code-Anonymous-black?logo=github)](https://anonymous.4open.science/r/adapt-or-not-vla/)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms)](LICENSE)

---

## Overview

Test-time adaptation (TTA) promises to improve vision-language-action (VLA) policies during deployment, but online weight updates can be harmful when the feedback driving adaptation is unreliable. We study this problem as **adaptation-worthiness**: deciding whether deployment-time feedback should be trusted enough to update a VLA policy.

We propose a lightweight **command–delta verification signal** that compares commanded Cartesian motion with observed proprioceptive displacement. This execution-level signal:
- Requires no shift labels, simulator state, success oracle, or additional neural network inference
- Suppresses online updates when command–delta mismatch is high (control authority loss)
- Preserves adaptation benefits when execution remains reliable

**Key finding:** Unconditional TTA helps under mild/appearance shifts but can degrade below the frozen baseline under severe hidden physics/control shifts. Our gate recovers those losses while preserving gains.

---

## Distribution Shift Evaluation Suite

We evaluate across 8 shift types at up to 5 severity levels each:

| Category | Shift | Severity range |
|----------|-------|---------------|
| **Appearance** | Gamma (brightness) | ±0.05 → ±0.25 |
| **Appearance** | Additive noise | σ=3 → σ=15 |
| **Appearance** | Gaussian blur | σ=0.4 → σ=2.0 |
| **Appearance** | Texture swap | Wall → Wall+Floor |
| **Physics** | Object weight | 10× → 200× |
| **Physics** | Gripper strength | 99% → 10% |
| **Control** | Action latency | 1 step → 8 steps |
| **Control** | Frequency drop | ÷2 → ÷16 |

Appearance shifts are **observable** (visible in camera frames). Physics and control shifts are **hidden** — they manifest only through proprioceptive state.

---

## Repository Structure

```
├── experiments/robot/libero/
│   ├── run_libero_eval.py          # Main evaluation loop (frozen / ttvla / robomonkey)
│   ├── verification_signals.py     # ← KEY CONTRIBUTION: blind gating mechanism
│   ├── run_shift_sweep.py          # Sweep all severities of one shift type
│   ├── libero_utils.py             # Shift application, LIBERO env helpers
│   └── perturbations.py            # Physics perturbations (object weight, gripper)
│
├── experiments/robot/
│   ├── openvla_utils.py            # OpenVLA logprob extraction
│   └── robomonkey_utils.py         # RoboMonkey action sampling
│
├── vla-scripts/
│   └── finetune_hdf5.py            # LoRA fine-tuning on shifted HDF5 demonstrations
│
└── prismatic/                      # OpenVLA base (unchanged from upstream)
```

---

## Installation

### One-command setup (recommended)

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://anonymous.4open.science/r/adapt-or-not-vla/
cd tta-worthiness/openvla

# 2. Run the setup script (creates conda env, installs all deps, downloads checkpoint)
bash setup.sh
```

`setup.sh` handles everything: conda environment, PyTorch, FlashAttention, LIBERO, dataset download, and the pretrained checkpoint. It saves the checkpoint to `checkpoints/` by default — override with `CHECKPOINT_DIR=/your/path bash setup.sh`.

### Manual setup (step by step)

<details>
<summary>Expand for manual installation steps</summary>

#### 1. Create environment

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
```

#### 2. Install PyTorch

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install OpenVLA

```bash
pip install -e .
pip install flash-attn==2.5.5 --no-build-isolation
```

#### 4. Install LIBERO

```bash
cd ../LIBERO
pip install -e .
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
cd ../openvla
```

#### 5. Install remaining dependencies

```bash
pip install -r experiments/robot/libero/libero_requirements.txt
pip install numpy==1.26.4 tensorflow==2.15.0
```

#### 6. (Optional) TT-VLA server for online adaptation

TT-VLA requires a separate process running the VLAC critic. See `../ttvla/tta.py` and source `../env.sh` to configure paths.

</details>

---

## Reproducing Paper Results (Table 3)

Set environment variables once before running any evaluation:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=.
CKPT=checkpoints/openvla-7b-finetuned-libero-spatial
```

The four conditions in Table 3 map directly to `run.sh` arguments:

```bash
# (1) Frozen — OpenVLA baseline, no adaptation
bash run.sh none object_weight 5

# (2) RoboMonkey — update-free test-time scaling baseline
bash run.sh robomonkey object_weight 5

# (3) TT-VLA unconditional — online adaptation, no gate
ENABLE_VERIFY=False bash run.sh ttvla object_weight 5

# (4) TT-VLA + Verif. (ours) — gated by ρ_cmd  ← main result
bash run.sh ttvla object_weight 5
```

> **TT-VLA server**: modes `ttvla` requires the VLAC critic server running in a separate terminal: `python ../ttvla/tta.py`

To run a different shift, pass `[shift_mode] [severity]`:
```bash
bash run.sh none latency 4
bash run.sh ttvla freq_drop 3
bash run.sh none gamma 2
```

---

## Running Evaluations

### `run.sh` — single condition

```
bash run.sh [mode] [shift_mode] [severity]
```

| Argument | Options | Default |
|----------|---------|---------|
| `mode` | `none` / `ttvla` / `robomonkey` | `none` |
| `shift_mode` | `gamma` `noise` `blur` `texture` `object_weight` `gripper_strength` `latency` `freq_drop` | `object_weight` |
| `severity` | `1`–`5` | `5` |

Override checkpoint or trial count via environment variables:
```bash
CHECKPOINT=/path/to/ckpt NUM_TRIALS=10 bash run.sh none object_weight 3
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `none` | `none` (frozen) / `ttvla` (TT-VLA adaptation) / `robomonkey` |
| `--shift_name` | — | `appearance` / `physics` / `control` |
| `--shift_mode` | — | `gamma`, `noise`, `blur`, `texture`, `object_weight`, `gripper_strength`, `latency`, `freq_drop` |
| `--severity` | `1` | Integer 1–5 (mild → severe) |
| `--enable_verification_signals` | `True` | Enable the command–delta gate |
| `--verify_cmd_mismatch_threshold` | `0.25` | τ: gate fires when ρ_cmd exceeds this (paper §3.4) |
| `--verify_window_size` | `20` | W: rolling window (timesteps) for ρ_cmd computation |
| `--num_trials_per_task` | `50` | Episodes per task (10 tasks → 500 total, matches paper) |

---

## Verification Signals Module

`experiments/robot/libero/verification_signals.py` implements the full gating system.

### Command–delta mismatch (key signal)

```python
from experiments.robot.libero.verification_signals import VerificationSignals

signals = VerificationSignals(shift_mode="object_weight", window_size=20)

# Inside the evaluation loop:
signals.update(img, action, log_probs, robot_state)
gate_ok, reason, signal_dict = signals.should_adapt(cfg)

if gate_ok:
    metrics = adapter.update(buffer, task_description, cfg)
else:
    print(f"[Gate] Skipping TTA: {reason}")
    # signal_dict["cmd_delta_sign_mismatch"] gives the current mismatch rate
```

The signal computes the fraction of timesteps where the commanded z-axis direction opposes the observed z-axis displacement:

```
mismatch_t = sign(cmd_z[t]) ≠ sign(Δz[t+1])   (only for |cmd_z| > threshold)
ρ_cmd = mean(mismatch_t) over window W
```

### Additional signals

| Signal | Type | Gate |
|--------|------|------|
| `cmd_delta_sign_mismatch` | Physics/Control | Primary gate (ρ_cmd > τ) |
| `brightness_deviation` | Appearance | Shift detection |
| `noise_level` | Appearance | Shift detection |
| `sharpness` | Appearance | Shift detection |
| `color_hist_shift` | Appearance | Shift detection |
| `gripper_error` | Physics | Supplementary |
| `action_repeat_ratio` | Control | Supplementary |
| `action_entropy` | All | Model uncertainty |
| `vlac_progress_slope` | TT-VLA | Adaptation quality |

---

## Evaluation Modes

| Mode | Description |
|------|-------------|
| `none` | Frozen policy (no adaptation) — baseline |
| `ttvla` | TT-VLA online adaptation with optional gate |
| `robomonkey` | RoboMonkey best-of-N sampling with optional gate |

---

## Acknowledgements

This codebase builds on:
- [OpenVLA](https://github.com/openvla/openvla) — base VLA model and training infrastructure
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — tabletop manipulation benchmark
- [TT-VLA](https://arxiv.org/abs/2512.14666) — progress-based test-time adaptation
- [RoboMonkey](https://github.com/RoboMonkeyRobots/RoboMonkey) — best-of-N sampling baseline
