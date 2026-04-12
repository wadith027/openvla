"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import subprocess
import time
import redis
import torch

from PIL import Image as PILImage

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from experiments.robot.robomonkey_utils import get_vla_action
from experiments.robot.libero.perturbations import apply_perturbation

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    SUPPORTED_APPEARANCE_MODES,
    SUPPORTED_PHYSICS_MODES,
    SUPPORTED_CONTROL_MODES,
    CONTROL_MAX_SEVERITY,
    SUPPORTED_SHIFT_MODES,
    SUPPORTED_SHIFT_NAMES,
    apply_physics_shift,
    apply_shift,
    build_episode_shift_state,
    build_control_shift_state,
    should_query_policy,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_physics_shift_value,
    PHYSICS_MAX_SEVERITY,
    resolve_physics_value,
    quat2axisangle,
    replace_target_textures,
    save_rollout_video,
    save_image,
)
from experiments.robot.openvla_utils import get_processor, get_logprob_of_action
from experiments.robot.robomonkey_utils import get_robomonkey_action
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
    get_action_policy
)

from experiments.robot.training_utils import OnlineAdapter


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    shift_name: str = "none"                         # Shift family. Options: none, appearance, physics, control
    shift_mode: str = "gamma"                        # Shift mode. Options (appearance): noise, blur, gamma, texture; (physics): object_weight, gripper_strength; (control): latency, freq_drop
    severity: int = 1                                # Shift severity (used when shift_name != none). Range: [1, 5]
    sweep_severity: Optional[int] = None             # Sweep severity label for plotting/aggregation. Range: [1, 5]
    physics_value_override: Optional[float] = None  # Override the severity-based physics multiplier directly (e.g. 1000.0)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = f"./experiments/log"        # Local directory for eval logs
    metrics_output_path: Optional[Union[str, Path]] = None  # Optional path for structured metrics JSON output

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    ###################################################################################################################
    # Online Adaptation / Inference Mode
    # mode: "none"       -- plain single-sample inference, no adaptation
    #        "ttvla"     -- TTA with OnlineAdapter (PPO) + TTVLA critic via Redis
    #        "robomonkey"-- best-of-N with RoboMonkey verifier, no weight updates
    #####################################################################################################################
    mode: str = "none"

    # TTA / TTVLA options (used when mode == "ttvla")
    ttvla_env: Optional[str] = 'ttvla'
    tta_step: int = 5

    # RoboMonkey options (used when mode == "robomonkey")
    initial_samples: int = 9
    augmented_samples: int = 32
    action_server_port: int = 3200
    reward_server_port: int = 3100
    task_id: int = 0
    transfer_dir: str = f"./transfer_images/{os.environ.get('SLURM_JOB_ID', 'local')}"

    
    # fmt: on


def validate_shift_config(cfg: GenerateConfig) -> None:
    """Validates shift config values and fails fast on invalid combinations."""
    if cfg.shift_name not in SUPPORTED_SHIFT_NAMES:
        raise ValueError(f"Unexpected shift_name '{cfg.shift_name}'. Supported values: {sorted(SUPPORTED_SHIFT_NAMES)}")
    if cfg.sweep_severity is not None:
        if not isinstance(cfg.sweep_severity, int) or not (1 <= cfg.sweep_severity <= 5):
            raise ValueError(f"Expected sweep_severity to be an integer in [1, 5], got: {cfg.sweep_severity}")
    if cfg.shift_name == "none":
        return
    if cfg.shift_name == "appearance":
        if cfg.shift_mode not in SUPPORTED_APPEARANCE_MODES:
            raise ValueError(
                f"shift_mode '{cfg.shift_mode}' is not valid for shift_name='appearance'. "
                f"Supported appearance modes: {sorted(SUPPORTED_APPEARANCE_MODES)}"
            )
        if not isinstance(cfg.severity, int) or not (1 <= cfg.severity <= 5):
            raise ValueError(f"Expected severity in [1, 5] for appearance shift, got: {cfg.severity}")
    elif cfg.shift_name == "physics":
        if cfg.shift_mode not in SUPPORTED_PHYSICS_MODES:
            raise ValueError(
                f"shift_mode '{cfg.shift_mode}' is not valid for shift_name='physics'. "
                f"Supported physics modes: {sorted(SUPPORTED_PHYSICS_MODES)}"
            )
        max_sev = PHYSICS_MAX_SEVERITY[cfg.shift_mode]
        if not isinstance(cfg.severity, int) or not (1 <= cfg.severity <= max_sev):
            raise ValueError(
                f"Expected severity in [1, {max_sev}] for physics shift_mode='{cfg.shift_mode}', got: {cfg.severity}"
            )
    elif cfg.shift_name == "control":
        if cfg.shift_mode not in SUPPORTED_CONTROL_MODES:
            raise ValueError(
                f"shift_mode '{cfg.shift_mode}' is not valid for shift_name='control'. "
                f"Supported control modes: {sorted(SUPPORTED_CONTROL_MODES)}"
            )
        max_sev = CONTROL_MAX_SEVERITY[cfg.shift_mode]
        if not isinstance(cfg.severity, int) or not (1 <= cfg.severity <= max_sev):
            raise ValueError(
                f"Expected severity in [1, {max_sev}] for control shift_mode='{cfg.shift_mode}', got: {cfg.severity}"
            )


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # OpenVLA README recommends center-crop for image-augmentation fine-tuned checkpoints.
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    validate_shift_config(cfg)

    # Set random seed. Note: small cross-machine variation may remain due to GPU nondeterminism.
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    os.makedirs(cfg.transfer_dir, exist_ok=True)

    # For other modes: load the model locally.
    model = None
    processor = None
    if cfg.mode != "robomonkey":
        model = get_model(cfg)

    # [OpenVLA] Resolve the action un-normalization key.
    if cfg.model_family == "openvla":
        if cfg.mode == "robomonkey":
            # Load dataset_statistics.json to resolve the key without loading the full model.
            ds_stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
            if os.path.isfile(ds_stats_path):
                with open(ds_stats_path) as f:
                    norm_stats = json.load(f)
                if cfg.unnorm_key not in norm_stats and f"{cfg.unnorm_key}_no_noops" in norm_stats:
                    cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        else:
            if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
                cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
            assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    if cfg.model_family == "openvla" and cfg.mode != "robomonkey":
        processor = get_processor(cfg)
    if cfg.mode == "ttvla":
        adapter = OnlineAdapter(model=model, processor=processor)
        model = adapter.model
    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    run_id += f"--shift={cfg.shift_name}"
    if cfg.shift_name != "none":
        run_id += f"--mode={cfg.shift_mode}--severity={cfg.severity}"
    if cfg.shift_name == "physics":
        run_id += f"--value={resolve_physics_value(cfg)}"
    if cfg.sweep_severity is not None:
        run_id += f"--sweep_s={cfg.sweep_severity}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.local_log_dir, cfg.shift_mode), exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, cfg.shift_mode, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    shift_cfg_str = (
        f"Shift config: shift_name={cfg.shift_name}, shift_mode={cfg.shift_mode}, "
        f"severity={cfg.severity}, sweep_severity={cfg.sweep_severity}"
    )
    print(shift_cfg_str)
    log_file.write(shift_cfg_str + "\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    if cfg.mode == "ttvla":
        script = Path(os.environ["TTA_SCRIPT"])
        socket_path = os.environ["TTA_SOCKET_PATH"]
        ready_token = f"{os.getpid()}-{time.time_ns()}"
        tta_env = os.environ.copy()
        tta_env["TTA_READY_TOKEN"] = ready_token
        tta_env["CUDA_VISIBLE_DEVICES"] = "1"

        python_exec = os.path.expandvars("$CONDA_ENVS_DIR/ttvla/bin/python")

        subprocess.Popen([
            python_exec,
            "-u",
            str(script)
        ], env=tta_env)

        while True:
            try:
                r = redis.Redis(unix_socket_path=socket_path)
                if r.get("tta:ready") == ready_token.encode():
                    break
            except redis.ConnectionError:
                time.sleep(0.5)

        r = redis.Redis(unix_socket_path=socket_path)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    per_task_metrics = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            if cfg.mode == "ttvla":
                adapter.reset_episode()
            if cfg.shift_name in ("none", "appearance"):
                shift_state = build_episode_shift_state(cfg, resize_size, task_id, episode_idx)
            elif cfg.shift_name == "physics":
                # Physics shift: appearance pipeline is fully disabled.
                # Images are passed to the model unmodified.
                shift_state = {"enabled": False}
                print("[shift] Appearance shift: DISABLED (physics mode — images unmodified)")
            else:
                # Control shift: appearance pipeline is fully disabled.
                # Images are passed to the model unmodified.
                assert cfg.shift_name == "control", f"Unexpected shift_name: {cfg.shift_name}"
                shift_state = {"enabled": False}
                print("[shift] Appearance shift: DISABLED (control mode — images unmodified)")

            texture_shift_info = None
            if shift_state["enabled"] and cfg.shift_mode == "texture":
                texture_shift_info = replace_target_textures(env, seed=shift_state["seed"], severity=cfg.severity)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # IMPORTANT: Apply physics perturbation AFTER set_init_state().
            # env.reset() / env.set_init_state() resets body_mass and actuator_gear,
            # so this must be re-applied each episode.
            if cfg.shift_name == "physics":
                apply_physics_shift(env, cfg)

            control_state = build_control_shift_state(cfg)

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            episode_images = []
            progress = []
            buffer = []
            last_action = None  # used to repeat last action under control shift (pre-latency / freq_drop)
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            if cfg.shift_name == "physics":
                physics_value = resolve_physics_value(cfg)
                episode_shift_str = (
                    f"Episode physics shift: task_id={task_id}, episode_idx={episode_idx}, "
                    f"mode={cfg.shift_mode}, severity={cfg.severity}, value={physics_value}"
                )
                print(episode_shift_str)
                log_file.write(episode_shift_str + "\n")
            elif cfg.shift_name == "control":
                episode_shift_str = (
                    f"Episode control shift: task_id={task_id}, episode_idx={episode_idx}, "
                    f"mode={cfg.shift_mode}, severity={cfg.severity}, value={control_state['value']}"
                )
                print(episode_shift_str)
                log_file.write(episode_shift_str + "\n")
            elif shift_state["enabled"]:
                if cfg.shift_mode == "texture":
                    swapped_count = 0 if texture_shift_info is None else texture_shift_info["swapped_material_count"]
                    target_count = 0 if texture_shift_info is None else texture_shift_info["target_material_count"]
                    episode_shift_str = (
                        "Episode shift params: "
                        f"task_id={task_id}, episode_idx={episode_idx}, seed={shift_state['seed']}, "
                        f"texture_swapped_materials={swapped_count}/{target_count}"
                    )
                else:
                    episode_shift_str = (
                        "Episode shift params: "
                        f"task_id={task_id}, episode_idx={episode_idx}, seed={shift_state['seed']}, "
                        f"gamma={shift_state['gamma']:.4f}, noise_std={shift_state['noise_std']:.4f}, "
                        f"blur_sigma={shift_state['blur_sigma']:.4f}"
                    )
                print(episode_shift_str)
                log_file.write(episode_shift_str + "\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    img = apply_shift(img, cfg, shift_state)

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    if cfg.mode == "robomonkey":
                            # use_wrist_image
                        if cfg.use_wrist_image:
                            wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image")
                            replay_wrist_images.append(wrist_img)

                        # buffering #obs_history images, optionally
                        image_history = replay_images[-cfg.obs_history :]
                        if len(image_history) < cfg.obs_history:
                            image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                        # same but for optional wrist images
                        if cfg.use_wrist_image:
                            wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                            if len(wrist_image_history) < cfg.obs_history:
                                wrist_image_history.extend(
                                    [replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history))
                                )
                            # interleaved images [... image_t, wrist_t ...]
                            image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

                        # Prepare observations dict
                        # Note: OpenVLA does not take proprio state as input
                        observation = {
                            "full_image": image_history,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        # Save raw obs image for reward model BEFORE TF preprocessing
                        raw_img = obs["agentview_image"][::-1, ::-1]
                        PILImage.fromarray(raw_img).save(os.path.join(cfg.transfer_dir, "reward_img.jpg"))

                        # Query RoboMonkey pipeline: sample actions from server, score with verifier, pick best
                        action = get_vla_action(
                            vla=None,
                            processor=None,
                            base_vla_name=cfg.model_family,
                            obs=observation,
                            task_label=task_description,
                            unnorm_key=cfg.unnorm_key,
                            center_crop=cfg.center_crop,
                            cfg=cfg,
                        )
                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        action = normalize_gripper_action(action, binarize=True)

                        if cfg.model_family in ["openvla", "prismatic"]:
                            action = invert_gripper_action(action)

                        # Log every 20 steps
                        if t % 20 == 0:
                            np.set_printoptions(precision=4, suppress=True)
                            print(f"  [t={t}] action: {action}")
                            log_file.write(f"  [t={t}] action: {action}\n")

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                        continue

                    else:
                        step = t - cfg.num_steps_wait
                        if should_query_policy(control_state, step, last_action is None):
                            action, action_tokens, log_probs = get_action_policy(
                                cfg,
                                model,
                                observation,
                                task_description,
                                processor=processor,
                                return_probs=True,
                            )
                            last_action = action
                        else:
                            action = last_action
                            action_tokens, log_probs = None, None
                    
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                    if cfg.mode == "ttvla":
                         # Get preprocessed image
                        img = get_libero_image(obs, resize_size)
                        img = apply_shift(img, cfg, shift_state)
                        observation_next = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }
                        if t >= cfg.num_steps_wait:
                            img_path = save_image(img, task_id, episode_idx, t, task_description, cfg.shift_mode, log_file)
                            episode_images.append(img_path)
                            r.rpush("tta_images", json.dumps({"obs": episode_images[-10:], "task_description": task_description}))

                            _, result_raw = r.blpop("tta_results", timeout=30)
                            result = json.loads(result_raw)
                            value_list = result["value_list"]

                            if len(value_list) > 0:
                                log_file.write(f"value_list[-1]: {value_list[-1]}\n")
                                log_file.flush()

                                p_t = value_list[-1]
                                r_t = p_t - progress[-1] if len(progress) > 0 else 0.0
                                progress.append(p_t)

                                entry = (observation, action_tokens, r_t, log_probs)
                                buffer.append(entry)
                        if t >= cfg.num_steps_wait and t % cfg.tta_step == 0 and len(buffer) > 0:
                            metrics = adapter.update(buffer, task_description, cfg)
                            if metrics is not None:
                                print("Loss" , metrics.get("loss", "No loss in metrics"))
                                log_file.write(f"Loss: {metrics.get('loss', 'No loss in metrics')}\n")
                                log_file.flush()
                            buffer = []
                                

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Free any cached GPU allocations between episodes to prevent fragmentation OOM
            torch.cuda.empty_cache()

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, shift=cfg.shift_mode, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        current_task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        current_total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        print(f"Current task success rate: {current_task_success_rate}")
        print(f"Current total success rate: {current_total_success_rate}")
        log_file.write(f"Current task success rate: {current_task_success_rate}\n")
        log_file.write(f"Current total success rate: {current_total_success_rate}\n")
        log_file.flush()
        per_task_metrics.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "task_episodes": task_episodes,
                "task_successes": task_successes,
                "task_success_rate": current_task_success_rate,
            }
        )
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": current_task_success_rate,
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    metrics_payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_id_note": cfg.run_id_note,
        "timestamp": DATE_TIME,
        "model_family": cfg.model_family,
        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
        "task_suite_name": cfg.task_suite_name,
        "seed": cfg.seed,
        "center_crop": cfg.center_crop,
        "num_trials_per_task": cfg.num_trials_per_task,
        "shift_name": cfg.shift_name,
        "shift_mode": cfg.shift_mode,
        "severity": cfg.severity,
        "sweep_severity": cfg.sweep_severity,
        "physics_value": resolve_physics_value(cfg) if cfg.shift_name == "physics" else None,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "per_task": per_task_metrics,
    }

    if cfg.metrics_output_path is not None:
        metrics_output_path = Path(cfg.metrics_output_path)
    else:
        metrics_output_path = Path(cfg.local_log_dir) / f"{run_id}.metrics.json"
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2, sort_keys=True)

    metrics_path_str = str(metrics_output_path)
    print(f"Saved metrics JSON at path {metrics_path_str}")
    log_file.write(f"Saved metrics JSON at path {metrics_path_str}\n")

    # Save local log file after all summary writes complete.
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": total_success_rate,
                "num_episodes/total": total_episodes,
                "summary/total_successes": total_successes,
                "summary/total_episodes": total_episodes,
                "summary/total_success_rate": total_success_rate,
                "summary/seed": cfg.seed,
                "summary/shift_name": cfg.shift_name,
                "summary/shift_mode": cfg.shift_mode,
                "summary/severity": cfg.severity,
                "summary/sweep_severity": -1 if cfg.sweep_severity is None else cfg.sweep_severity,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
