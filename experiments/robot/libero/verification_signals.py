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
_ACTION_TO_METERS         = 0.05   # approximate m/step per unit of normalized XYZ action


class VerificationSignals:
    """
    Accumulates per-step signals and produces a gate decision for TTA.

    All signals are blind: they use only images, actions, log-probs, and robot
    state — no knowledge of the configured shift_mode or severity is used in
    the gate decision.  (shift_mode is stored only for the `.signals.jsonl` log.)
    """

    LOG_EVERY: int = 10  # print formatted summary every N timesteps

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

        # Colour-histogram baseline: EMA that tracks gradual scene changes
        # (robot arm motion) but spikes on sudden texture swaps.
        # alpha=0.98 → half-life ≈ 34 steps; a texture swap causes an
        # immediate large L1 delta that the EMA cannot absorb in one step.
        self._hist_ema_alpha      : float                = 0.98
        self._hist_ema_baseline   : Optional[np.ndarray] = None

        # Sharpness / noise baselines (accumulated over first window_size frames)
        self._baseline_sharpness  : Optional[float]      = None
        self._baseline_noise      : Optional[float]      = None
        self._n_baseline_frames   : int                  = 0

        # Persistence filter: Gate 1 only fires after this many consecutive
        # steps above the severity threshold.  Filters single-frame spikes
        # (e.g. brief hist_shift outlier) without masking real sustained shifts.
        self._severity_persist_required: int = 3
        self._severity_consecutive     : int = 0  # steps above threshold so far

        # Accumulative progress estimation (ttvla mode, paper §3.3).
        # v_i = v_{i-1} + (1 - v_{i-1}) * c_t  where c_t ∈ [-1, 1] from VLAC.
        # Compares current frame to the most recent milestone, accumulates with
        # diminishing returns so the value stays in [0, 1] and is noise-resistant.
        self._acc_progress : float        = 0.0
        self._acc_history  : deque[float] = deque(maxlen=10)  # v_t at each update

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
        # EMA colour-histogram baseline: initialise on first frame, then
        # slowly track the scene so gradual robot-arm drift is absorbed.
        curr_hist = _color_histogram(img)
        if self._hist_ema_baseline is None:
            self._hist_ema_baseline = curr_hist.copy()
        else:
            a = self._hist_ema_alpha
            self._hist_ema_baseline = a * self._hist_ema_baseline + (1.0 - a) * curr_hist
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

    def update_accumulative_progress(self, c_t: float) -> None:
        """
        Apply the paper's accumulative progress formula (§3.3, eq. 2, scaled to [0, 1]).

        c_t ∈ [-1, 1]: incremental VLAC value comparing the current frame to the
                       most recent milestone frame (positive = progress, negative = regression).

        v_i = v_{i-1} + (1 - v_{i-1}) * c_t

        Diminishing returns: positive progress pushes v toward 1 by a fraction of the
        remaining gap; negative critics reduce v proportionally.  Clamped to [0, 1].

        Call this once per VLAC query (every tta_step steps in ttvla mode).
        """
        self._acc_progress = self._acc_progress + (1.0 - self._acc_progress) * float(c_t)
        self._acc_progress = max(0.0, min(1.0, self._acc_progress))
        self._acc_history.append(self._acc_progress)

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
        s["color_hist_shift"]     = _color_hist_shift(img, self._hist_ema_baseline)
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
        if len(self._actions) >= 2 and len(self._robot_states) >= 2:
            s["cmd_delta_sign_mismatch"] = _cmd_delta_sign_mismatch(
                list(self._actions), list(self._robot_states)
            )
            s["cmd_xyz_weighted_mismatch"] = _cmd_xyz_weighted_mismatch(
                list(self._actions), list(self._robot_states)
            )
            s["gripper_close_fail"] = _gripper_close_fail_ratio(
                list(self._actions), list(self._robot_states)
            )
        if len(self._actions) >= 2:
            s["action_oscillation"] = _action_oscillation_ratio(list(self._actions))
        if len(self._actions) >= 2 and len(self._robot_states) >= 2:
            s["motion_suppression"] = _motion_suppression_ratio(  # logged only, not scored
                list(self._actions), list(self._robot_states)
            )

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

        # ── Accumulative progress (ttvla only, paper §3.3) ────────────────────
        if self._acc_history:
            s["acc_progress"] = self._acc_progress
        if len(self._acc_history) >= 3:
            s["acc_progress_slope"] = _linear_slope(list(self._acc_history))

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

        Single gate: ρ_cmd (command–delta sign mismatch) compared against
        cfg.verify_cmd_mismatch_threshold (default 0.25, per paper §3.4 Eq. 7).
        Blocks the TTA weight update when ρ_cmd > τ.
        """
        self._n_tta_opportunities += 1
        signals = self.compute()

        # Wait for window to fill before trusting the signal
        if self._n_steps < self.window_size:
            reason = f"warming_up ({self._n_steps}/{self.window_size} steps)"
            self._n_tta_skipped += 1
            return False, reason, signals

        # Single gate — ρ_cmd (paper Eq. 6–7): XYZ magnitude-weighted mismatch
        tau = getattr(cfg, "verify_cmd_mismatch_threshold", 0.25)
        rho = signals.get("cmd_xyz_weighted_mismatch", 0.0)
        if rho > tau:
            reason = f"rho_cmd={rho:.3f} > tau={tau}"
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

        # ── Severity sub-score breakdown ──────────────────────────────────────
        # Visual sub-score: worst of brightness/noise/blur/texture
        vis_parts = []
        if "brightness_deviation" in signals:
            vis_parts.append(min(signals["brightness_deviation"] / _BRIGHTNESS_MAX_DEVIATION, 1.0))
        if "noise_level" in signals:
            vis_parts.append(min(max(signals["noise_level"] - _NOISE_BASELINE_STD, 0.0) / _NOISE_SEVERITY_RANGE, 1.0))
        if "sharpness" in signals:
            vis_parts.append(min(max(1.0 - signals["sharpness"] / _SHARPNESS_REF, 0.0), 1.0))
        if "color_hist_shift" in signals:
            vis_parts.append(min(signals["color_hist_shift"] / _HIST_SHIFT_MAX, 1.0))
        visual_sub = max(vis_parts) if vis_parts else 0.0

        # Physics sub-score: worst of action_repeat / cmd_mismatch
        phy_parts = []
        if "action_repeat_ratio" in signals:
            phy_parts.append(signals["action_repeat_ratio"])
        if "cmd_delta_sign_mismatch" in signals:
            phy_parts.append(min(max(signals["cmd_delta_sign_mismatch"] - 0.10, 0.0) / 0.25, 1.0))
        phy_sub = max(phy_parts) if phy_parts else 0.0

        # Uncertainty sub-score
        unc_sub = 0.0
        if "action_entropy" in signals:
            unc_sub = min(max(signals["action_entropy"] - _ENTROPY_NORMAL_MAX, 0.0) / 2.0, 1.0)

        lines = [
            f"┌─ {header} {fill}┐",
            f"│  Shift Severity : bright_dev={_f('brightness_deviation')}"
            f"  sharp={_f('sharpness',1)}"
            f"  noise={_f('noise_level',2)}",
            f"│                   gripper_err={_f('gripper_error')}"
            f"  act_repeat={_f('action_repeat_ratio',0,pct=True)}"
            f"  hist_shift={_f('color_hist_shift',3)}"
            f"  cmd_mismatch={_f('cmd_delta_sign_mismatch',0,pct=True)}"
            f"  xyz_mismatch={_f('cmd_xyz_weighted_mismatch',0,pct=True)}"
            f"  grip_fail={_f('gripper_close_fail',2)}"
            f"  oscillation={_f('action_oscillation',0,pct=True)}"
            f"  motion_supp={_f('motion_suppression',2)}",
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
        if "acc_progress" in signals:
            acc_v = signals.get("acc_progress", 0.0)
            lines.append(
                f"│  Accum Progress : v={_f('acc_progress',3)} [{_bar(acc_v)}]"
                f"  slope={_f('acc_progress_slope',4)}"
                f"  (n={len(self._acc_history)} updates)"
            )
        lines += [
            f"│  Score Breakdown: visual={visual_sub:.3f} [{_bar(visual_sub,5)}]"
            f"  physics={phy_sub:.3f} [{_bar(phy_sub,5)}]"
            f"  uncert={unc_sub:.3f} [{_bar(unc_sub,5)}]",
            f"│  Severity Score : {_f('severity_score',3)} / 1.0  [{_bar(sev)}]"
            f"  consec={self._severity_consecutive}  skip_rate={self._n_tta_skipped}/{self._n_tta_opportunities}",
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


def _motion_suppression_ratio(
    actions     : list[np.ndarray],
    robot_states: list[np.ndarray],
    cmd_threshold: float = 0.15,
) -> float:
    """
    Fraction of commanded motion that the robot fails to execute.

    For each step where the policy issues a non-trivial XYZ command, compare
    the expected displacement (action_mag * _ACTION_TO_METERS) to the actual
    displacement (||delta robot_state[:3]||).  Returns the mean suppression
    ratio over the window:

        suppression_i = 1 - actual_mag / expected_mag   (clamped to [0, 1])

    Near 0 during normal operation; near 1 when the robot is commanded to move
    but barely does (high freq_drop, or robot physically stuck).
    """
    n = min(len(actions), len(robot_states)) - 1
    if n < 1:
        return 0.0
    total = 0.0
    valid = 0
    for i in range(n):
        cmd_mag = float(np.linalg.norm(actions[i][:3]))
        if cmd_mag < cmd_threshold:
            continue
        actual_mag = float(np.linalg.norm(
            robot_states[i + 1][:3] - robot_states[i][:3]
        ))
        expected_mag = cmd_mag * _ACTION_TO_METERS
        total += max(0.0, 1.0 - actual_mag / expected_mag)
        valid += 1
    return min(total / valid, 1.0) if valid > 0 else 0.0


def _action_oscillation_ratio(
    actions   : list[np.ndarray],
    threshold : float = 0.1,
) -> float:
    """
    Fraction of consecutive command pairs that reverse direction on XYZ axes.

    With high control latency the robot overshoots, corrects, and overshoots
    again, producing oscillatory sign-flips in its commands.  Near-zero commands
    (|v| < threshold) are ignored because direction is ambiguous there.

    Returns a value in [0, 1].  Background ~5-10%; values >25% indicate
    oscillation consistent with high latency.
    """
    n = len(actions) - 1
    if n < 1:
        return 0.0
    flips = 0
    total = 0
    for i in range(n):
        for dim in range(3):
            prev = float(actions[i][dim])
            curr = float(actions[i + 1][dim])
            if abs(prev) > threshold and abs(curr) > threshold:
                total += 1
                if prev * curr < 0:
                    flips += 1
    return float(flips) / total if total > 0 else 0.0


def _gripper_close_fail_ratio(
    actions     : list[np.ndarray],
    robot_states: list[np.ndarray],
    close_threshold : float = 0.5,
    lift_threshold  : float = 0.001,
) -> float:
    """
    Fraction of steps where the gripper is commanded close but the arm is not rising.

    A successful grasp: robot closes gripper → arm rises (carrying the object).
    A failed grasp:     robot closes gripper → arm stays flat or descends.

    For gripper_strength shift the fingers physically close (qpos→0) but grip force
    is insufficient to hold the object, so the arm never lifts after the close command.

    close_threshold : action[-1] < this → commanded close (raw policy output, 0=close)
    lift_threshold  : minimum delta_z (m/step) to count as a successful lift
    Returns a value in [0, 1].
    """
    n = min(len(actions), len(robot_states)) - 1
    if n < 1:
        return 0.0
    count = 0
    total = 0
    for i in range(n):
        if float(actions[i][-1]) >= close_threshold:
            continue                          # not commanding close
        total += 1
        delta_z = float(robot_states[i + 1][2]) - float(robot_states[i][2])
        if delta_z < lift_threshold:          # arm not rising after close command
            count += 1
    return float(count) / total if total > 0 else 0.0


def _cmd_delta_sign_mismatch(
    actions     : list[np.ndarray],
    robot_states: list[np.ndarray],
    cmd_threshold: float = 0.3,
) -> float:
    """
    Fraction of (action[t], state[t+1]-state[t]) pairs where cmd_z and delta_z
    have opposite signs.

    cmd_z   = action[t][2]                              (policy z-command, normalised [-1,1])
    delta_z = robot_state[t+1][2] - robot_state[t][2]  (actual z movement, metres)

    A high ratio means the arm is moving opposite to the commanded direction —
    a reliable proxy for control authority loss / robot confusion after a failed grasp.
    Background noise level: ~8-10%.  Values >30% indicate a failing episode.
    """
    n = min(len(actions), len(robot_states)) - 1  # need consecutive state pairs
    if n < 1:
        return 0.0
    count = 0
    valid = 0
    for i in range(n):
        cmd_z = float(actions[i][2])
        if abs(cmd_z) < cmd_threshold:
            continue                               # near-zero command → ambiguous direction, skip
        delta_z = float(robot_states[i + 1][2]) - float(robot_states[i][2])
        valid += 1
        if cmd_z * delta_z < 0:                   # opposite signs → mismatch
            count += 1
    return float(count) / valid if valid > 0 else 0.0


def _cmd_xyz_weighted_mismatch(
    actions     : list[np.ndarray],
    robot_states: list[np.ndarray],
    cmd_threshold: float = 0.3,
) -> float:
    """
    XYZ magnitude-weighted sign mismatch between commanded and actual movement.

    Extends _cmd_delta_sign_mismatch to all three spatial axes and weights each
    pair by the magnitude of the command, so large confident direction commands
    that reverse count more than borderline ones.

        score = sum(|cmd_d| for mismatched dims) / sum(|cmd_d| for all valid dims)

    Background noise ~5-8%.  Values >25% indicate sustained control authority loss.
    """
    n = min(len(actions), len(robot_states)) - 1
    if n < 1:
        return 0.0
    total_weight = 0.0
    mismatch_weight = 0.0
    for i in range(n):
        for dim in range(3):
            cmd = float(actions[i][dim])
            if abs(cmd) < cmd_threshold:
                continue
            delta = float(robot_states[i + 1][dim]) - float(robot_states[i][dim])
            w = abs(cmd)
            total_weight += w
            if cmd * delta < 0:
                mismatch_weight += w
    return mismatch_weight / total_weight if total_weight > 0.0 else 0.0


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
        # Always use global clean-frame constant — episode baseline is contaminated
        # because noise shift is applied from frame 1, so baseline ≈ current level.
        excess = max(signals["noise_level"] - _NOISE_BASELINE_STD, 0.0)
        visual.append(min(excess / _NOISE_SEVERITY_RANGE, 1.0))
    if "sharpness" in signals:
        # Always use global clean-frame reference — episode baseline is contaminated
        # because blur is applied from frame 1, so baseline ≈ current sharpness.
        blur_score = max(1.0 - signals["sharpness"] / _SHARPNESS_REF, 0.0)
        visual.append(min(blur_score, 1.0))
    if "color_hist_shift" in signals:
        visual.append(min(signals["color_hist_shift"] / _HIST_SHIFT_MAX, 1.0))
    if visual:
        components.append(max(visual))

    # Physics / control severity
    # NOTE: gripper_error is intentionally excluded from severity_score.
    # A commanded-vs-actual gripper mismatch of 1.0 is normal during any grasp
    # (robot commands close while gripper is still open) and is not a reliable
    # indicator of distribution shift severity for non-physics tasks.
    # It is still logged as a raw signal in .signals.jsonl for analysis.
    phy_ctrl = []
    if "action_repeat_ratio" in signals:
        phy_ctrl.append(signals["action_repeat_ratio"])          # already [0, 1]
    if "cmd_delta_sign_mismatch" in signals:
        # Background noise ~8-10%.  Map [10%, 35%+] → [0, 1].
        raw = signals["cmd_delta_sign_mismatch"]
        phy_ctrl.append(min(max(raw - 0.10, 0.0) / 0.25, 1.0))
    if "gripper_close_fail" in signals:
        phy_ctrl.append(signals["gripper_close_fail"])
    if "action_oscillation" in signals:
        # Background ~5-10%.  Map [10%, 35%+] → [0, 1].
        raw = signals["action_oscillation"]
        phy_ctrl.append(min(max(raw - 0.10, 0.0) / 0.25, 1.0))
    if "cmd_xyz_weighted_mismatch" in signals:
        # Background ~5-8%.  Map [8%, 30%+] → [0, 1].
        raw = signals["cmd_xyz_weighted_mismatch"]
        phy_ctrl.append(min(max(raw - 0.08, 0.0) / 0.22, 1.0))
    # motion_suppression excluded from score — miscalibrated (_ACTION_TO_METERS off)
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
