"""
verification_signals.py

Blind observation-based verification signals that gate whether test-time adaptation
(TTA) should proceed for a given episode step.  "Blind" means signals are inferred
solely from images, robot state, and model outputs — the shift type and severity are
deliberately NOT used so the module generalises to real-world deployment.

Supported shift types (all detected without oracle knowledge):
  Appearance : gamma, noise, blur, texture
  Physics    : object_weight, gripper_strength
  Control    : latency, freq_drop

Usage (inside the eval loop):
    signals = VerificationSignals(shift_mode=cfg.shift_mode,
                                  window_size=cfg.verify_window_size)
    # --- inside timestep loop ---
    signals.update(img, action, log_probs, robot_state)
    signals.update_vlac(p_t)                        # after VLAC result arrives
    # --- at TTA update trigger ---
    gate_ok, reason, signal_dict = signals.should_adapt(cfg)
    if gate_ok:
        metrics = adapter.update(...)
        signals.record_tta_update(metrics["loss"], vlac_before, vlac_after)
    # --- every LOG_EVERY steps ---
    print(signals.format_summary(t, episode_idx, task_id, gate_ok, reason, signal_dict))
"""

from __future__ import annotations

import json
from collections import deque
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Calibrated normalization ranges derived from libero_utils.py severity tables.
# These are used to map raw signal values into [0, 1].
_BRIGHTNESS_MAX_DEVIATION = 0.22   # gamma_offset=0.25 → ~22% mean shift
_NOISE_BASELINE_STD       = 18.0   # clean Laplacian std for 256×256 frames
_NOISE_SEVERITY_RANGE     = 35.0   # additional std at max noise_std=15.0
_SHARPNESS_REF            = 600.0  # Laplacian var of a clean, unblurred frame
_HIST_SHIFT_MAX           = 0.45   # L1 distance for full texture swap
_GRIPPER_MAX_QPOS         = 0.045  # LIBERO gripper finger range ~[0, 0.045]
_ENTROPY_NORMAL_MAX       = 2.5    # nats; typical well-conditioned policy


class VerificationSignals:
    """
    Accumulates per-step signals and produces a gate decision for TTA.

    All signals are blind: they use only images, actions, log-probs, and robot
    state — no knowledge of the configured shift_mode or severity is used in
    the gate decision.  (shift_mode is stored only for the `.signals.jsonl` log.)
    """

    LOG_EVERY: int = 20  # print formatted summary every N timesteps

    def __init__(self, shift_mode: str, window_size: int = 10) -> None:
        self.shift_mode  = shift_mode
        self.window_size = window_size

        # Rolling windows
        self._images      : deque[np.ndarray] = deque(maxlen=window_size)
        self._actions     : deque[np.ndarray] = deque(maxlen=window_size)
        self._log_probs   : deque[float]       = deque(maxlen=window_size)
        self._robot_states: deque[np.ndarray] = deque(maxlen=window_size)
        self._vlac_values : deque[float]       = deque(maxlen=window_size)

        # Shorter windows for adaptation quality
        self._tta_losses         : deque[float] = deque(maxlen=5)
        self._vlac_pre_post_deltas: deque[float] = deque(maxlen=5)

        # First-frame baselines (accumulated over first window_size frames)
        self._first_frame_hist    : Optional[np.ndarray] = None
        self._baseline_sharpness  : Optional[float]      = None  # for relative blur scoring
        self._baseline_noise      : Optional[float]      = None  # for relative noise scoring
        self._n_baseline_frames   : int                  = 0

        # Episode-level counters
        self._n_steps            = 0
        self._n_tta_opportunities = 0
        self._n_tta_skipped       = 0

        # Last computed signal dict (cached, refreshed on compute())
        self._last_signals: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Data ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        img        : np.ndarray,          # (H, W, 3) uint8
        action     : np.ndarray,          # (7,) raw action from policy, gripper in [0,1]
        log_probs  : Optional[float],     # sequence_log_prob (negative scalar) or None
        robot_state: np.ndarray,          # (8,) = [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
    ) -> None:
        """Call once per active timestep, after get_action_policy()."""
        self._images.append(img.astype(np.float32))
        self._actions.append(action.copy())
        self._robot_states.append(robot_state.copy())
        if log_probs is not None:
            self._log_probs.append(float(log_probs))
        if self._first_frame_hist is None:
            self._first_frame_hist = _color_histogram(img)
        # Accumulate baseline sharpness/noise over first window_size frames
        if self._n_baseline_frames < self.window_size:
            s = _sharpness(img)
            n = _noise_level(img)
            if self._baseline_sharpness is None:
                self._baseline_sharpness = s
                self._baseline_noise     = n
            else:
                # Running mean
                k = self._n_baseline_frames + 1
                self._baseline_sharpness = self._baseline_sharpness + (s - self._baseline_sharpness) / k
                self._baseline_noise     = self._baseline_noise     + (n - self._baseline_noise)     / k
            self._n_baseline_frames += 1
        self._n_steps += 1

    def update_vlac(self, vlac_value: float) -> None:
        """Call after receiving the VLAC critic result."""
        self._vlac_values.append(float(vlac_value))

    def record_tta_update(
        self,
        loss       : float,
        vlac_before: float,
        vlac_after : float,
    ) -> None:
        """Call after a successful TTA weight update."""
        self._tta_losses.append(float(loss))
        self._vlac_pre_post_deltas.append(float(vlac_after) - float(vlac_before))

    # ──────────────────────────────────────────────────────────────────────────
    # Signal computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        """Return a flat dict of all signal values from the current window."""
        s: dict = {}
        if not self._images:
            return s

        img = self._images[-1]  # most recent frame

        # ── Appearance signals ────────────────────────────────────────────────
        s["brightness_deviation"] = _brightness_deviation(img)
        s["sharpness"]            = _sharpness(img)
        s["noise_level"]          = _noise_level(img)
        s["color_hist_shift"]     = _color_hist_shift(img, self._first_frame_hist)
        # Pass baselines into signals so severity_score can compute relative drops
        if self._baseline_sharpness is not None and self._baseline_sharpness > 1e-6:
            s["_baseline_sharpness"] = self._baseline_sharpness
        if self._baseline_noise is not None:
            s["_baseline_noise"] = self._baseline_noise

        # ── Physics / control signals ─────────────────────────────────────────
        if self._actions and self._robot_states:
            s["gripper_error"]       = _gripper_error(self._actions[-1], self._robot_states[-1])
        if len(self._actions) >= 2:
            s["action_repeat_ratio"] = _action_repeat_ratio(list(self._actions))

        # ── Model uncertainty ─────────────────────────────────────────────────
        if self._log_probs:
            lp_list = list(self._log_probs)
            s["action_log_prob_mean"] = float(np.mean(lp_list))
            # Proxy entropy: -mean(log_prob).  More negative log_prob = higher entropy.
            s["action_entropy"]       = float(-np.mean(lp_list))
        if len(self._actions) >= 2:
            s["action_l2_variance"] = _action_variance(list(self._actions))

        # ── VLAC adaptation quality ───────────────────────────────────────────
        if len(self._vlac_values) > 0:
            s["vlac_last"] = float(self._vlac_values[-1])
        if len(self._vlac_values) >= 3:
            s["vlac_progress_slope"] = _linear_slope(list(self._vlac_values))
        if self._vlac_pre_post_deltas:
            s["vlac_pre_post_delta_mean"] = float(np.mean(list(self._vlac_pre_post_deltas)))
        if len(self._tta_losses) >= 2:
            s["tta_loss_trend"] = _linear_slope(list(self._tta_losses))

        # ── Aggregated severity score [0, 1] ──────────────────────────────────
        s["severity_score"] = _severity_score(s)

        self._last_signals = s
        return s

    # ──────────────────────────────────────────────────────────────────────────
    # Gate decision
    # ──────────────────────────────────────────────────────────────────────────

    def should_adapt(self, cfg) -> tuple[bool, str, dict]:
        """
        Returns (should_adapt: bool, reason: str, signals: dict).

        For ttvla mode all four gates are active.
        For robomonkey mode VLAC-based gates are skipped (no critic signal).
        """
        self._n_tta_opportunities += 1
        signals = self.compute()

        sev = signals.get("severity_score", 0.0)
        ent = signals.get("action_entropy",  0.0)

        # Gate 1 — observation severity (all modes)
        if sev > cfg.verify_severity_threshold:
            reason = f"severity_score={sev:.3f} > {cfg.verify_severity_threshold}"
            self._n_tta_skipped += 1
            return False, reason, signals

        # Gate 2 — model entropy (all modes)
        if ent > cfg.verify_entropy_threshold:
            reason = f"entropy={ent:.2f} nats > {cfg.verify_entropy_threshold}"
            self._n_tta_skipped += 1
            return False, reason, signals

        # Gates 3 & 4 — adaptation quality (ttvla only)
        if getattr(cfg, "mode", "none") == "ttvla":
            delta = signals.get("vlac_pre_post_delta_mean", None)
            if delta is not None and len(self._vlac_pre_post_deltas) >= 2:
                if delta < cfg.verify_vlac_delta_threshold:
                    reason = f"vlac_Δ={delta:.3f} < {cfg.verify_vlac_delta_threshold} (2+ updates)"
                    self._n_tta_skipped += 1
                    return False, reason, signals

            slope = signals.get("vlac_progress_slope", None)
            if slope is not None and slope < cfg.verify_vlac_slope_threshold:
                reason = f"vlac_slope={slope:.4f} < {cfg.verify_vlac_slope_threshold}"
                self._n_tta_skipped += 1
                return False, reason, signals

        return True, "ok", signals

    # ──────────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ──────────────────────────────────────────────────────────────────────────

    def format_summary(
        self,
        t          : int,
        episode_idx: int,
        task_id    : int,
        gate_ok    : bool,
        reason     : str,
        signals    : Optional[dict] = None,
    ) -> str:
        """Return a formatted multi-line summary block for stdout / log_file."""
        if signals is None:
            signals = self._last_signals
        W = 76  # box width

        def _f(key, digits=3, pct=False):
            v = signals.get(key)
            if v is None:
                return "N/A"
            if pct:
                return f"{v * 100:.0f}%"
            return f"{v:.{digits}f}"

        def _bar(v, width=10):
            v = max(0.0, min(float(v), 1.0))
            filled = round(v * width)
            return "▓" * filled + "░" * (width - filled)

        sev    = signals.get("severity_score", 0.0)
        sym    = "✓ ADAPT" if gate_ok else "✗ SKIP "
        header = f"[VerifySignals t={t:04d} ep={episode_idx} task={task_id} shift={self.shift_mode}]"
        fill   = "─" * max(0, W - len(header) - 4)

        lines = [
            f"┌─ {header} {fill}┐",
            f"│  Shift Severity : bright_dev={_f('brightness_deviation')}"
            f"  sharp={_f('sharpness',1)}"
            f"  noise={_f('noise_level',2)}",
            f"│                   gripper_err={_f('gripper_error')}"
            f"  act_repeat={_f('action_repeat_ratio',0,pct=True)}"
            f"  hist_shift={_f('color_hist_shift',3)}",
            f"│  Model Uncert.  : entropy={_f('action_entropy',2)} nats"
            f"  logprob={_f('action_log_prob_mean',2)}"
            f"  act_var={_f('action_l2_variance',4)}",
        ]
        if "vlac_last" in signals or "vlac_progress_slope" in signals:
            lines.append(
                f"│  VLAC Progress  : slope={_f('vlac_progress_slope',4)}"
                f"  last={_f('vlac_last',3)}"
                f"  pre_post_Δ={_f('vlac_pre_post_delta_mean',3)}"
                f"  loss_trend={_f('tta_loss_trend',4)}"
            )
        lines += [
            f"│  Severity Score : {_f('severity_score',3)} / 1.0  [{_bar(sev)}]"
            f"  skip_rate={self._n_tta_skipped}/{self._n_tta_opportunities}",
            f"│  Gate Decision  : {sym}  ({reason})",
            "└" + "─" * (W - 1) + "┘",
        ]
        return "\n".join(lines)

    def format_timestep_record(
        self,
        t          : int,
        episode_idx: int,
        task_id    : int,
        gate_ok    : bool,
        reason     : str,
        signals    : Optional[dict] = None,
    ) -> str:
        """Return a JSON line for the .signals.jsonl file."""
        if signals is None:
            signals = self._last_signals
        record = {
            "t"            : t,
            "episode"      : episode_idx,
            "task_id"      : task_id,
            "shift_mode"   : self.shift_mode,
            "gate_decision": gate_ok,
            "gate_reason"  : reason,
        }
        record.update({k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in signals.items()})
        return json.dumps(record)

    def get_episode_summary(self) -> dict:
        """Return a compact dict summarising signals over the episode (for metrics JSON)."""
        signals = self.compute()
        lp = list(self._log_probs)
        vlac = list(self._vlac_values)
        return {
            "final_severity_score"  : signals.get("severity_score"),
            "mean_action_entropy"   : float(-np.mean(lp)) if lp else None,
            "mean_vlac_progress"    : float(np.mean(vlac)) if vlac else None,
            "vlac_progress_slope"   : signals.get("vlac_progress_slope"),
            "n_tta_opportunities"   : self._n_tta_opportunities,
            "n_tta_skipped"         : self._n_tta_skipped,
            "tta_skip_rate"         : (self._n_tta_skipped / self._n_tta_opportunities
                                       if self._n_tta_opportunities > 0 else None),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Pure signal functions (no class state, easy to unit-test)
# ──────────────────────────────────────────────────────────────────────────────

def _laplacian(img: np.ndarray) -> np.ndarray:
    """2-D discrete Laplacian of the grayscale version of img (H×W×3 or H×W)."""
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float32)
    # 4-neighbour finite-difference Laplacian (fast, no scipy dependency)
    lap = (
        -gray[:-2, 1:-1]
        - gray[2:,  1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
        + 4.0 * gray[1:-1, 1:-1]
    )
    return lap


def _brightness_deviation(img: np.ndarray) -> float:
    """Normalised absolute deviation of mean pixel intensity from 128."""
    return float(abs(img.mean() - 128.0) / 128.0)


def _sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — low value indicates blur."""
    return float(np.var(_laplacian(img)))


def _noise_level(img: np.ndarray) -> float:
    """Std-dev of the Laplacian — high value indicates added noise."""
    return float(np.std(_laplacian(img)))


def _color_histogram(img: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Normalised per-channel histogram concatenated into a 1-D vector."""
    hists = []
    for c in range(img.shape[2] if img.ndim == 3 else 1):
        channel = img[:, :, c] if img.ndim == 3 else img
        h, _ = np.histogram(channel, bins=n_bins, range=(0.0, 255.0))
        h = h / (h.sum() + 1e-8)
        hists.append(h)
    return np.concatenate(hists)


def _color_hist_shift(
    img          : np.ndarray,
    baseline_hist: Optional[np.ndarray],
) -> float:
    """L1 distance between current and first-frame color histograms."""
    if baseline_hist is None:
        return 0.0
    curr = _color_histogram(img)
    return float(np.sum(np.abs(curr - baseline_hist)))


def _gripper_error(action: np.ndarray, robot_state: np.ndarray) -> float:
    """
    Absolute error between commanded and observed gripper position.

    action[-1]        : commanded gripper in [0, 1] (raw policy output)
    robot_state[6:8]  : two-finger joint positions in [0, GRIPPER_MAX_QPOS]
    """
    commanded = float(action[-1])                            # [0, 1]
    # Normalise actual qpos to [0, 1]
    actual = float(np.mean(robot_state[6:8])) / _GRIPPER_MAX_QPOS
    actual = np.clip(actual, 0.0, 1.0)
    return float(abs(commanded - actual))


def _action_repeat_ratio(actions: list[np.ndarray]) -> float:
    """Fraction of consecutive steps where the action is (nearly) identical."""
    if len(actions) < 2:
        return 0.0
    n_repeat = sum(
        1 for i in range(1, len(actions))
        if np.allclose(actions[i], actions[i - 1], atol=1e-4)
    )
    return float(n_repeat) / (len(actions) - 1)


def _action_variance(actions: list[np.ndarray]) -> float:
    """Mean per-dimension variance of the action window."""
    arr = np.stack(actions)           # (T, action_dim)
    return float(np.mean(np.var(arr, axis=0)))


def _linear_slope(values: list[float]) -> float:
    """Slope of the least-squares line through the value sequence."""
    x = np.arange(len(values), dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _severity_score(signals: dict) -> float:
    """
    Aggregate [0, 1] severity score.

    Computed as the maximum over four independent sub-scores so that any single
    severely corrupted domain (visual / physics / control / uncertainty) is
    caught even if the others look normal.
    """
    components: list[float] = []

    # Visual severity: worst of brightness, noise, blur, texture
    visual = []
    if "brightness_deviation" in signals:
        visual.append(min(signals["brightness_deviation"] / _BRIGHTNESS_MAX_DEVIATION, 1.0))
    if "noise_level" in signals:
        # Use episode baseline when available; otherwise fall back to global constant.
        baseline = signals.get("_baseline_noise", _NOISE_BASELINE_STD)
        excess = max(signals["noise_level"] - baseline, 0.0)
        visual.append(min(excess / _NOISE_SEVERITY_RANGE, 1.0))
    if "sharpness" in signals:
        # Use relative drop from episode baseline when available (adaptive to scene texture).
        # Falls back to absolute _SHARPNESS_REF for the first window.
        ref = signals.get("_baseline_sharpness", _SHARPNESS_REF)
        if ref > 1e-6:
            blur_score = max(1.0 - signals["sharpness"] / ref, 0.0)
            visual.append(min(blur_score, 1.0))
    if "color_hist_shift" in signals:
        visual.append(min(signals["color_hist_shift"] / _HIST_SHIFT_MAX, 1.0))
    if visual:
        components.append(max(visual))

    # Physics / control severity
    phy_ctrl = []
    if "gripper_error" in signals:
        phy_ctrl.append(min(signals["gripper_error"] / 0.6, 1.0))
    if "action_repeat_ratio" in signals:
        phy_ctrl.append(signals["action_repeat_ratio"])          # already [0, 1]
    if phy_ctrl:
        components.append(max(phy_ctrl))

    # Model uncertainty severity
    if "action_entropy" in signals:
        excess = max(signals["action_entropy"] - _ENTROPY_NORMAL_MAX, 0.0)
        components.append(min(excess / 2.0, 1.0))

    if not components:
        return 0.0
    # Take the max so any severely disrupted domain triggers the gate
    return float(max(components))
