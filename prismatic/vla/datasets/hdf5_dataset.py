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
        demo_root_dir: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ):
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.samples = []
        self.episodes = []          # (hdf5_path, T, task_id) one per episode
        self.sample_episode_idx = []  # parallel to self.samples
        all_actions = []

        episode_idx = 0
        for task_id in range(10):
            task_dir = os.path.join(demo_root_dir, f"task_{task_id:02d}")
            if not os.path.exists(task_dir):
                continue
            for hdf5_path in sorted(glob.glob(os.path.join(task_dir, "*.hdf5"))):
                with h5py.File(hdf5_path, "r") as f:
                    actions = f["data/demo_0/actions"][:]
                    T = actions.shape[0]
                    all_actions.append(actions)
                    for t in range(T):
                        if actions[t, -1] < 0.5:
                            continue
                        self.samples.append((hdf5_path, t, task_id))
                        self.sample_episode_idx.append(episode_idx)
                self.episodes.append((hdf5_path, T, task_id))
                episode_idx += 1

        base_stats_path = "/projects/bgub/models/openvla/openvla-7b-finetuned-libero-spatial/dataset_statistics.json"
        with open(base_stats_path) as f:
            base_stats = json.load(f)
        self.dataset_statistics = {"libero_spatial": base_stats["libero_spatial"]}

        print(f"HDF5ShiftedDemoDataset: {len(self.samples)} steps from {demo_root_dir}")
        print(f"  Episodes: {len(self.episodes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, t, task_id = self.samples[idx]
        instruction = LIBERO_SPATIAL_TASK_DESCRIPTIONS[task_id]

        with h5py.File(hdf5_path, "r") as f:
            img_np = f["data/demo_0/obs/agentview_rgb"][t]
            action = f["data/demo_0/actions"][t].astype(np.float32)

        img_np = img_np[::-1, ::-1].copy()
        image = Image.fromarray(img_np)

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)