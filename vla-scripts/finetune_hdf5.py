import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
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
    vla_path: str = "openvla/openvla-7b"

    # Directory Paths
    demo_root_dir: str = ""
    dataset_name: str = "shifted_demos"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # Fine-tuning Parameters
    batch_size: int = 16
    max_steps: int = 5_000
    save_steps: int = 1000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = False
    save_latest_checkpoint_only: bool = True

    # LoRA Arguments
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    # Validation / Early Stopping
    val_fraction: float = 0.2        # fraction of episodes held out for validation
    val_steps: int = 50              # how often to run validation
    early_stopping_patience: int = 5 # number of val checks with no improvement before stopping
    val_seed: int = 42               # seed for episode split reproducibility

    # Tracking Parameters
    wandb_project: str = "openvla"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    use_wandb: bool = False
    run_id_note: Optional[str] = None


def run_validation(vla, val_dataloader, device_id):
    """Runs one pass over the val set and returns mean loss."""
    vla.eval()
    total_loss = 0.0
    n_steps = 0
    with torch.no_grad():
        for batch in val_dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
            total_loss += output.loss.item()
            n_steps += 1
    vla.train()
    return total_loss / max(n_steps, 1)


def save_checkpoint(vla, processor, run_dir, adapter_dir, vla_path, use_lora, distributed_state, step):
    """Saves merged checkpoint."""
    if distributed_state.is_main_process:
        print(f"Saving Model Checkpoint for Step {step}")
        save_dir = adapter_dir if use_lora else run_dir
        processor.save_pretrained(run_dir)
        vla.module.save_pretrained(save_dir)

    dist.barrier()

    if use_lora:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(run_dir)
            print(f"Saved Model Checkpoint for Step {step} at: {run_dir}")

    dist.barrier()


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    assert cfg.demo_root_dir != "", "Must provide --demo_root_dir!"
    assert torch.cuda.is_available()

    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

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

    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

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

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Build full dataset
    vla_dataset = HDF5ShiftedDemoDataset(
        cfg.demo_root_dir,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    # Episode-level train/val split
    n_episodes = len(vla_dataset.episodes)
    n_val_eps = max(1, int(n_episodes * cfg.val_fraction))
    n_train_eps = n_episodes - n_val_eps

    rng = np.random.default_rng(cfg.val_seed)
    shuffled_eps = rng.permutation(n_episodes)
    val_ep_set = set(shuffled_eps[:n_val_eps].tolist())
    train_ep_set = set(shuffled_eps[n_val_eps:].tolist())

    train_indices = [i for i, ep in enumerate(vla_dataset.sample_episode_idx) if ep in train_ep_set]
    val_indices = [i for i, ep in enumerate(vla_dataset.sample_episode_idx) if ep in val_ep_set]

    train_dataset = Subset(vla_dataset, train_indices)
    val_dataset = Subset(vla_dataset, val_indices)

    print(f"Split: {n_train_eps} train episodes ({len(train_indices)} steps) | "
          f"{n_val_eps} val episodes ({len(val_indices)} steps)")

    # Save dataset statistics (from full dataset — loaded from fixed base path, no leak)
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    if distributed_state.is_main_process and cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    best_val_loss = float("inf")
    patience_counter = 0
    best_step = 0

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        batch_idx = 0
        done = False

        while not done:
            for batch in train_dataloader:
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
                    print(f"  [step={gradient_step_idx}] train_loss={smoothened_loss:.4f} "
                          f"acc={smoothened_action_accuracy:.4f} l1={smoothened_l1_loss:.4f}")
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

                # Validation + early stopping
                if gradient_step_idx > 0 and gradient_step_idx % cfg.val_steps == 0:
                    val_loss = run_validation(vla, val_dataloader, device_id)

                    if distributed_state.is_main_process:
                        print(f"  [step={gradient_step_idx}] val_loss={val_loss:.4f} "
                              f"(best={best_val_loss:.4f} @ step {best_step})")
                        if cfg.use_wandb:
                            wandb.log({"val_loss": val_loss}, step=gradient_step_idx)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = gradient_step_idx
                            patience_counter = 0
                            print(f"  New best val loss — saving checkpoint")
                            save_checkpoint(
                                vla, processor, run_dir, adapter_dir,
                                cfg.vla_path, cfg.use_lora, distributed_state,
                                gradient_step_idx
                            )
                        else:
                            patience_counter += 1
                            print(f"  No improvement ({patience_counter}/{cfg.early_stopping_patience})")
                            if patience_counter >= cfg.early_stopping_patience:
                                print(f"Early stopping at step {gradient_step_idx}. "
                                      f"Best val loss {best_val_loss:.4f} at step {best_step}.")
                                done = True
                                break

                if gradient_step_idx >= cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached!")
                    # Save final if it happens to be best
                    if distributed_state.is_main_process and best_step == 0:
                        save_checkpoint(
                            vla, processor, run_dir, adapter_dir,
                            cfg.vla_path, cfg.use_lora, distributed_state,
                            gradient_step_idx
                        )
                    done = True
                    break

                batch_idx += 1


if __name__ == "__main__":
    finetune()