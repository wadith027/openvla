import os, json, math, requests
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
from experiments.robot.token_action_converter import TokenActionConverter
from experiments.robot.openvla_utils import OPENVLA_V01_SYSTEM_PROMPT, crop_and_resize

_converter = None

def _get_converter(unnorm_key, pretrained_checkpoint):
    global _converter
    if _converter is None:
        _converter = TokenActionConverter(unnorm_key=unnorm_key, pretrained_checkpoint=pretrained_checkpoint)
    return _converter
os.makedirs("./transfer_images", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_robomonkey_action(obs, task_label, cfg, vla=None, processor=None):
    """
    RoboMonkey best of N action selection
    """
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if cfg.center_crop:
        image = _apply_center_crop(image)

    # Save image for servers (they take absolute paths)
    img_path = os.path.abspath(f"./transfer_images/obs_{cfg.task_id}.jpg")
    image.save(img_path)

    # 1. Sample initial_samples 
    if vla is not None and processor is not None:
        output_ids, actions = _get_local_vla_batch_actions(vla, processor, image, task_label, cfg.unnorm_key, cfg.initial_samples, cfg.pretrained_checkpoint)
    else:
        output_ids, actions = _get_batch_actions(task_label, img_path, cfg)

    output_ids, actions = _preprocess(output_ids, actions)
    
    if len(actions) == 1:
        return actions[0]

    # 2. Gaussian augmentation around the batch mean/variance
    aug_ids, aug_actions = _augment(actions, cfg.augmented_samples, cfg.unnorm_key, cfg.pretrained_checkpoint)

    # 3. Score with reward model
    rewards = _get_rewards(task_label, img_path, aug_ids, cfg)

    best_idx = np.argmax(rewards)
    print(f"[robomonkey] n_valid={len(actions)} rewards: min={min(rewards):.3f} max={max(rewards):.3f} mean={np.mean(rewards):.3f} | best={rewards[best_idx]:.3f} action={np.round(aug_actions[best_idx], 4)}", flush=True)

    return aug_actions[best_idx]


def _get_local_vla_batch_actions(vla, processor, image, task_label, unnorm_key, n_samples, base_vla_name, temperature=1.0):
    """ Samples N sample actions in a single forward pass of the VLA """
    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: "
            f"What action should the robot take to {task_label.lower()}?\n Out:"
            )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\n Out:"
    
    single = processor(prompt, image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

    if not torch.all(single["input_ids"][:, -1] == 29871):
        extra = torch.tensor([[29871]], device=single["input_ids"].device, dtype=single["input_ids"].dtype)     
        single["input_ids"] = torch.cat([single["input_ids"], extra], dim=1)                                    
        single["attention_mask"] = torch.cat(
            [single["attention_mask"], torch.ones((1, 1), device=single["attention_mask"].device, dtype=single["attention_mask"].dtype)],dim=1,                                                                                              
        )  

    # Duplicate inputs for batch processing
    inputs = {k: v.expand(n_samples, *v.shape[1:]).clone() for k, v in single.items()}

    with torch.inference_mode():
        outputs = vla.generate(
            **inputs,
            max_new_tokens=vla.get_action_dim(unnorm_key),
            do_sample=True,
            temperature=temperature,
        )

    action_dim = vla.get_action_dim(unnorm_key)
    action_token_ids = outputs[:, -action_dim:].cpu().numpy()

    # tokens -> actions
    discretized = vla.vocab_size - action_token_ids
    discretized = np.clip(discretized - 1, 0, vla.bin_centers.shape[0] - 1)
    normalized = vla.bin_centers[discretized]

    stats = vla.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])

    actions = np.where(mask, 0.5 * (normalized + 1) * (hi - lo) + lo, normalized)

    return action_token_ids, actions

def _get_batch_actions(instruction, image_path, cfg):
    ids, acts = [], []
    for _ in range(cfg.initial_samples):
        r = requests.post(
            f"http://127.0.0.1:{cfg.action_server_port}/batch",
            json={"instruction": instruction.lower(),
                "image_path": image_path,
                "batch_size": 1, "temperature": 1.0},
            timeout=30,
        ).json()
        ids.extend(r["output_ids"])
        acts.extend(r["actions"])
    return np.array(ids), np.array(acts)


def _preprocess(output_ids, actions):
    mask = np.all((output_ids >= 31744) & (output_ids <= 32000), axis=1)
    return output_ids[mask], actions[mask]


def _augment(actions, n_samples, unnorm_key, pretrained_checkpoint):
    mean = np.mean(actions, axis=0)
    std  = np.sqrt(np.var(actions, axis=0))
    aug  = np.random.normal(mean, std, size=(n_samples, 7))
    # Gripper is binary
    aug[:, -1] = float(mean[-1] >= 0.5)
    # Clip to valid action ranges from model norm_stats
    converter = _get_converter(unnorm_key, pretrained_checkpoint)
    action_stats = converter.norm_stats[unnorm_key]["action"]
    lo = np.array(action_stats["q01"])
    hi = np.array(action_stats["q99"])
    aug[:, :-1] = np.clip(aug[:, :-1], lo[:-1], hi[:-1])
    ids = np.array([converter.action_to_token(a) for a in aug])
    return ids, aug


def _get_rewards(instruction, image_path, action_ids, cfg):
    rewards = []
    bs = 2  # reduce to 1 if OOM
    for i in range(math.ceil(len(action_ids) / bs)):
        batch = action_ids[i*bs:(i+1)*bs].tolist()
        r = requests.post(
            f"http://127.0.0.1:{cfg.reward_server_port}/process",
            json={"instruction": instruction.lower(),
                "image_path": image_path,
                "action": batch},
            timeout=60,
        ).json()
        rewards.extend(r["rewards"])
    return rewards



def _apply_center_crop(image):
    crop_scale = 0.9
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size=1)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")



def get_batch_actions(instruction, image_path, batch_size, temperature=1.0, cfg=None):
    """Get multiple action predictions from the VLA action server."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    payload = {
        "instructions": [instruction] * batch_size,
        "image_path": image_path,
        "temperature": temperature,
    }
    response = requests.post(
        f"http://127.0.0.1:{cfg.action_server_port}/batch",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    if response.status_code != 200:
        raise Exception(f"Error from action server: {response.text}")
    data = response.json()
    return np.array(data["output_ids"]), np.array(data["actions"])


def preprocess_actions(output_ids, actions):
    output_ids = np.array(output_ids)
    output_ids = np.where(output_ids == 31745, 31744, output_ids)
    actions = np.array(actions)
    mask = np.all((output_ids >= 31744) & (output_ids <= 32000), axis=1)
    return output_ids[mask], actions[mask]


def get_unique_actions(output_ids, actions):
    output_ids = np.array(output_ids)
    actions = np.array(actions)
    _, indices = np.unique(output_ids, axis=0, return_index=True)
    indices = sorted(indices)
    return output_ids[indices], actions[indices]


def generate_augmented_samples_from_batch(batch_actions, num_samples, unnorm_key, model_name_or_path):
    mean = np.mean(batch_actions, axis=0)
    var = np.var(batch_actions, axis=0)
    augmented = np.random.normal(mean, np.sqrt(var), size=(num_samples, 7))
    gripper_p = float(np.clip(mean[-1], 0.0, 1.0))
    augmented[:, -1] = np.random.binomial(1, gripper_p, size=num_samples).astype(float)
    converter = _get_converter(unnorm_key, model_name_or_path)
    stats = converter.norm_stats[unnorm_key]["action"]
    lo, hi = np.array(stats["q01"]), np.array(stats["q99"])
    augmented[:, :-1] = np.clip(augmented[:, :-1], lo[:-1], hi[:-1])
    ids = np.array([converter.action_to_token(a) for a in augmented])
    return ids, augmented


def get_rewards(instruction, image_path, action_ids, cfg):
    rewards = []
    bs = 2
    for i in range(math.ceil(len(action_ids) / bs)):
        batch = action_ids[i * bs:(i + 1) * bs].tolist()
        r = requests.post(
            f"http://127.0.0.1:{cfg.reward_server_port}/process",
            json={"instruction": instruction.lower(), "image_path": image_path, "action": batch},
            timeout=60,
        ).json()
        rewards.extend(r["rewards"])
    return rewards


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, cfg=None):
    """Generates an action with the VLA policy."""

    # only supports 1 image
    if isinstance(obs["full_image"], list):
        obs["full_image"] = obs["full_image"][0]

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

        # Save processed image and path for inference
        transfer_dir = cfg.transfer_dir
        os.makedirs(transfer_dir, exist_ok=True)
        image_path = os.path.join(transfer_dir, "vla_processed_img.jpg")
        image.save(image_path)

    # Get initial action samples from VLA Serving Engine
    instruction = task_label.lower()
    image_path = str(Path(os.path.join(cfg.transfer_dir, "vla_processed_img.jpg")).absolute())
    # Uncomment to debug instruction/image path:
    # print(f"  [VLA] instruction: '{instruction}'")
    # print(f"  [VLA] image_path:  {image_path}  exists={os.path.exists(image_path)}")
    output_ids, actions = get_batch_actions(
        instruction=instruction,
        image_path=image_path,
        batch_size=cfg.initial_samples,
        temperature=1,
        cfg=cfg
    )

    # Preprocess initial actions
    if cfg.initial_samples == 1 and cfg.augmented_samples == 1:
        print(f"  [VLA] baseline action (no verifier): {actions[0]}")
        return actions[0]
    output_ids, actions = preprocess_actions(output_ids, actions)
    _, unique = get_unique_actions(output_ids, actions)
    if len(unique)==1:
        return unique[0]

    # Generate augmented samples based on the mean and variance of a batch of actions.
    output_ids, actions = generate_augmented_samples_from_batch(
        batch_actions=actions,
        num_samples=cfg.augmented_samples,
        unnorm_key=cfg.unnorm_key,
        model_name_or_path=cfg.pretrained_checkpoint,
    )

    # Score each action with robomonkey verifier
    output_ids, actions = get_unique_actions(output_ids, actions)
    reward_img = str(Path(os.path.join(cfg.transfer_dir, "reward_img.jpg")).absolute())
    rewards = get_rewards(instruction, reward_img, output_ids, cfg=cfg)

    selected_index = np.argmax(rewards)

    return actions[selected_index]