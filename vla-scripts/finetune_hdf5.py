"""
finetune_hdf5.py

LoRA fine-tuning of OpenVLA on HDF5 shifted demonstrations (Oracle upper bound baseline).

Run with:
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_hdf5.py \
        --vla_path <CHECKPOINT_PATH> \
        --demo_root_dir <PATH/TO/shifted_demos/latency/sev_1> \
        --dataset_name latency_sev1 \
        --run_root_dir <PATH/TO/LOGS/DIR>
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.hdf5_dataset import HDF5ShiftedDemoDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    demo_root_dir: str = ""                                         # Path to HDF5 shifted demos dir (e.g. experiments/shifted_demos/latency/sev_1)
    dataset_name: str = "shifted_demos"                             # Name tag for this run
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 5_000                                          # Max number of fine-tuning steps
    save_steps: int = 1000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = False                                         # Whether to train with image augmentations
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to
    wandb_entity: str = "YOUR_WANDB_ENTITY"                         # Name of entity to log under
    use_wandb: bool = False                                         # Whether to use W&B logging
    run_id_note: Optional[str] = None                               # Extra note for logging

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    assert cfg.demo_root_dir != "", "Must provide --demo_root_dir!"

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # LoRA
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # DDP
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load HDF5 Dataset
    vla_dataset = HDF5ShiftedDemoDataset(
        cfg.demo_root_dir,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    # Save Dataset Statistics
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )

    # W&B
    if distributed_state.is_main_process and cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        batch_idx = 0
        done = False

        while not done:
            for batch in dataloader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                # Metrics
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

                if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                    print(f"  [step={gradient_step_idx}] loss={smoothened_loss:.4f} acc={smoothened_action_accuracy:.4f} l1={smoothened_l1_loss:.4f}")
                    if cfg.use_wandb:
                        wandb.log(
                            {
                                "train_loss": smoothened_loss,
                                "action_accuracy": smoothened_action_accuracy,
                                "l1_loss": smoothened_l1_loss,
                            },
                            step=gradient_step_idx,
                        )

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()

                # Save checkpoint
                if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                        save_dir = adapter_dir if cfg.use_lora else run_dir
                        processor.save_pretrained(run_dir)
                        vla.module.save_pretrained(save_dir)

                    dist.barrier()

                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            merged_vla.save_pretrained(run_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")

                    dist.barrier()

                if gradient_step_idx >= cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    done = True
                    break

                batch_idx += 1


if __name__ == "__main__":
    finetune()