"""
perturbations.py

Injectable physics-based distribution shift perturbations for LIBERO environments.

IMPORTANT: env.reset() / env.set_init_state() resets sim.model.body_mass and
actuator_gear back to their original values. This function must be called AFTER
env.set_init_state() inside the episode loop, not once before it.

Usage:
    from experiments.robot.libero.perturbations import apply_perturbation
    obs = env.set_init_state(initial_states[episode_idx])
    apply_perturbation(env, perturbation_type="object_weight", perturbation_value=10.0)
"""


def apply_perturbation(env, perturbation_type: str, perturbation_value: float):
    """Apply a physics distribution shift perturbation to the LIBERO environment.

    Args:
        env: LIBERO OffScreenRenderEnv instance
        perturbation_type: "object_weight" | "gripper_strength" | "none"
        perturbation_value: multiplier applied to the selected property
    """
    if perturbation_type == "none":
        return
    elif perturbation_type == "object_weight":
        _apply_object_weight(env, perturbation_value)
    elif perturbation_type == "gripper_strength":
        _apply_gripper_strength(env, perturbation_value)
    else:
        raise ValueError(
            f"Unknown perturbation_type: '{perturbation_type}'. "
            "Choose from: 'none', 'object_weight', 'gripper_strength'."
        )


BOWL_KEYWORDS = ["bowl", "akita"]  # "akita" is the turbosquid brand name for LIBERO's bowl object


def _apply_object_weight(env, multiplier: float):
    """Multiply the mass of all bowl bodies by multiplier."""
    model = env.sim.model
    matched = []
    for i in range(model.nbody):
        name = model.body_id2name(i).lower()
        if any(kw in name for kw in BOWL_KEYWORDS):
            prev = float(model.body_mass[i])
            model.body_mass[i] *= multiplier
            new = float(model.body_mass[i])
            matched.append((name, prev, new))
            print(f"[perturbation] object_weight x{multiplier} | {name}: {prev} -> {new}")
    if not matched:
        raise RuntimeError(
            "No bowl bodies found in sim model. "
            f"Available bodies: {[model.body_id2name(i) for i in range(model.nbody)]}"
        )


def _apply_gripper_strength(env, multiplier: float):
    """Multiply the gear ratio of all gripper/finger actuators by multiplier."""
    model = env.sim.model
    matched = []
    for i in range(model.nu):
        name = model.actuator_id2name(i)
        if "gripper" in name.lower() or "finger" in name.lower():
            prev = float(model.actuator_gear[i, 0])
            model.actuator_gear[i, 0] *= multiplier
            new = float(model.actuator_gear[i, 0])
            matched.append((name, prev, new))
            print(f"[perturbation] gripper_strength x{multiplier} | {name}: {prev} -> {new}")
    if not matched:
        raise RuntimeError(
            "No gripper/finger actuators found in sim model. "
            f"Available actuators: {[model.actuator_id2name(i) for i in range(model.nu)]}"
        )
