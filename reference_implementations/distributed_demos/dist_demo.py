from __future__ import annotations

import functools
import math
import os
import sys
from typing import Any

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler, set_seed
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from data import get_dataloaders
from utils import cleanup, load_model_and_tokenizer, setup


def shard_model(
    model: nn.Module,
    layer_to_wrap: type[nn.Module],
    strategy: str,
) -> nn.Module:
    """Shard the model to workers using FSDP.

    Args:
    ----
        model: The model to be sharded.
        layer_to_wrap: The layer we are wrapping using FSDP.
        strategy: The sharding strategy to use.

    Returns:
    -------
        The sharded module with the requested configurations.

    """
    fsdp_cfg = {}

    # enable mixed-precision training
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    fsdp_cfg["mixed_precision"] = mp_policy

    # wrap the correct layer of the model
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_to_wrap},
    )

    # retrieve the FSDP sharding strategy
    sharding_strategy = getattr(ShardingStrategy, strategy)

    fsdp_cfg["auto_wrap_policy"] = transformer_wrap_policy
    fsdp_cfg["sharding_strategy"] = sharding_strategy
    fsdp_cfg["device_id"] = torch.cuda.current_device()

    if dist.get_rank() == 0:
        print(f"FSDP config: {fsdp_cfg}")
    model = FSDP(model, **fsdp_cfg)
    print(
        "Model sharded. Per device model parameters are ",
        f"{sum(p.numel() for p in model.parameters())}",
    )

    return model

def get_steps_per_epoch(
    original_length: int,
    gas: int,
    epochs: int,
) -> tuple[int, int]:
    """Calculate steps for weight updates."""
    sharded_ds_orig_len = math.ceil(
        original_length / dist.get_world_size(),
    )
    num_update_steps_per_epoch = max(
        sharded_ds_orig_len // gas,
        1,
    )
    max_steps = math.ceil(
        epochs * num_update_steps_per_epoch,
    )
    return max_steps, num_update_steps_per_epoch

def print_main(*args: list[Any], **kwargs: dict[str, Any]) -> None:
    """Print only on the main rank."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def train_step(
    batch: dict[str, torch.Tensor],
    step: int,
    gas: int,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
) -> float:
    """Step training once.

    Args:
    ----
        batch: The batch of data.
        step: The current training step.
        gas: The number of gradient accumulation steps.
        model: The model.
        optimizer: The optimizer.
        lr_scheduler: The scheduler.

    Returns:
    -------
        The train loss.

    """
    batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
    batch["labels"] = batch["labels"].type(torch.LongTensor)

    if (step + 1) % gas != gas - 1:
        # no need to sync while accumulating gradients
        with model.no_sync():
            out = model(**batch)
            tr_step_loss = out.loss
            (tr_step_loss / gas).backward()
            # clips gradient norm to avoid exploding gradients
            if isinstance(model, FSDP):
                model.clip_grad_norm_(1.0)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    else:
        # next forward / backward pass will be synced
        dist.barrier()
        out = model(**batch)
        tr_step_loss = out.loss
        (tr_step_loss / gas).backward()
        if isinstance(model, FSDP):
            model.clip_grad_norm_(1.0)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return _gather(tr_step_loss.reshape(1)).mean().item()

def eval_step(
    eval_dl: DataLoader,
    model: nn.Module,
    step: int,
) -> float:
    """Complete evaluation once.

    Args:
    ----
        eval_dl: The evaluation dataloader.
        model: The model.
        step: The current training step.

    Returns:
    -------
        The test loss.

    """
    model.eval()
    eval_loss = torch.tensor(0.0).to(torch.cuda.current_device())
    for _, batch in enumerate(eval_dl):
        with torch.no_grad():
            batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
            batch["labels"] = batch["labels"].type(torch.LongTensor)
            out = model(**batch)
            eval_loss += out.loss

    gathered_eval_loss = _gather(eval_loss.reshape(1)).mean().item()
    mean_eval_loss = gathered_eval_loss / len(eval_dl)

    print_main(f"Step: {step}, eval loss: {mean_eval_loss}")

    model.train()
    return gathered_eval_loss

def _gather(x: torch.Tensor) -> torch.Tensor:
    output_tensors = [x.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, x)
    return torch.cat(output_tensors, dim=0)

def main() -> None:
    """Execute the main loop."""
    # set CUDA related dependencies
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    model_path = "/ssd003/projects/aieng/finetuning_bootcamp/downloads/tinyllama/"  # noqa: E501
    dataset_path = "/ssd003/projects/aieng/finetuning_bootcamp/downloads/newsmediabias/debiased_profainty_check_with_keywords.csv"  # noqa: E501
    batch_size = 8  # use batches of powers of 2
    epochs = 1
    # aiming for a global batch size of 128:
    gas = 128 // (batch_size * world_size) # gradient accumulation steps.
    lr = 2.0e-5
    eval_steps = 25  # how often to eval
    dist_type = "DDP"  # either DDP or FSDP
    assert dist_type in ("DDP", "FSDP")

    # set a seed
    set_seed(42)

    print(f"Rank: {rank}, World size: {world_size}")
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        use_mp=True,
        use_fa=True,
    )

    # load datasets
    train_dl, test_dl, orig_length = get_dataloaders(
        dataset_path,
        tokenizer,
        batch_size,
    )

    # get number of model parameter update steps
    max_steps, num_update_steps_per_epoch = get_steps_per_epoch(
        orig_length,
        gas,
        epochs,
    )

    if dist_type == "FSDP":
        # shard model with FSDP
        model = shard_model(model, LlamaDecoderLayer, "FULL_SHARD")
    else:
        # wrap model with DDP
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])

    # get optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1.0e-5,
    )

    # get scheduler
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        math.ceil(0.05 * num_update_steps_per_epoch),
        max_steps,
    )

    # execute training
    step = 0
    for _ in range(epochs):
        model.train()
        train_dl_iterator = iter(train_dl)
        for _ in tqdm(
            range(len(train_dl)),
            disable=rank != 0,
            file=sys.__stdout__,
        ):
            batch = next(train_dl_iterator)
            train_step(
                batch,
                step,
                gas,
                model,
                optimizer,
                scheduler,
            )
            if step % eval_steps == 0:
                eval_step(test_dl, model, step)
            step += 1

    # save the trained model
    if dist_type == "FSDP":
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            if rank == 0:
                torch.save(model_state, "model_weights.pt")
    else:
        model_state = model.state_dict()
        if rank == 0:
            torch.save(model_state, "model_weights.pt")

if __name__ == "__main__":
    setup()
    main()
    cleanup()
