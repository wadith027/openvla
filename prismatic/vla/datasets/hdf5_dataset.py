import glob
import json
import os
from pathlib import Path
from typing import Type

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer

IGNORE_INDEX = -100

LIBERO_SPATIAL_TASK_DESCRIPTIONS = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl on the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
}


class HDF5ShiftedDemoDataset(Dataset):
    def __init__(
        self,
        demo_root_dir: str,                      # e.g. experiments/shifted_demos/latency/sev_1
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ):
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Collect all (hdf5_path, step_idx, task_id) tuples
        self.samples = []
        all_actions = []

        for task_id in range(10):
            task_dir = os.path.join(demo_root_dir, f"task_{task_id:02d}")
            if not os.path.exists(task_dir):
                continue
            for hdf5_path in sorted(glob.glob(os.path.join(task_dir, "*.hdf5"))):
                with h5py.File(hdf5_path, "r") as f:
                    actions = f["data/demo_0/actions"][:]  # (T, 7)
                    T = actions.shape[0]
                    all_actions.append(actions)
                    for t in range(T):
                        self.samples.append((hdf5_path, t, task_id))

        # Compute action statistics for de-normalization
        all_actions = np.concatenate(all_actions, axis=0)  # (N, 7)
        q01 = np.quantile(all_actions, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(all_actions, 0.99, axis=0).astype(np.float32)

        self.dataset_statistics = {
            "shifted_demos": {
                "action": {"q01": q01, "q99": q99}
            }
        }

        print(f"HDF5ShiftedDemoDataset: {len(self.samples)} steps from {demo_root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, t, task_id = self.samples[idx]
        instruction = LIBERO_SPATIAL_TASK_DESCRIPTIONS[task_id]

        with h5py.File(hdf5_path, "r") as f:
            img_np = f["data/demo_0/obs/agentview_rgb"][t]  # (224, 224, 3) uint8
            action = f["data/demo_0/actions"][t].astype(np.float32)  # (7,)

        img_np = img_np[::-1, ::-1].copy() # rotate 180° to match train preprocessing
        image = Image.fromarray(img_np)

        # Build prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # Only compute loss on action tokens
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)