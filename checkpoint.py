import os
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


CHECKPOINT_VERSION = 1


def get_random_state() -> dict:
    random_state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }

    if torch.cuda.is_available():
        random_state["cuda"] = torch.cuda.get_rng_state_all()

    return random_state


def restore_random_state(random_state: Optional[dict]) -> None:
    if not random_state:
        return

    python_state = random_state.get("python")
    torch_state = random_state.get("torch")
    cuda_state = random_state.get("cuda")

    if python_state is not None:
        random.setstate(python_state)

    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu())

    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def get_model_metadata(model: nn.Module) -> dict:
    metadata = {
        "model_class": model.__class__.__name__,
    }

    if hasattr(model, "bert"):
        bert = model.bert

        if hasattr(bert, "config"):
            metadata["bert_config"] = bert.config.to_dict()

    return metadata


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    global_step: int,
    best_dev_macro_f1: float,
    train_config: dict,
    dev_metrics: dict,
    early_stopping_counter: int,
) -> None:
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "best_dev_macro_f1": best_dev_macro_f1,
        "train_config": dict(train_config),
        "dev_metrics": dict(dev_metrics),
        "model_metadata": get_model_metadata(model),
        "random_state": get_random_state(),
        "early_stopping_counter": early_stopping_counter,
    }

    directory = os.path.dirname(
        os.path.abspath(filepath)
    )
    os.makedirs(directory, exist_ok=True)

    temporary_filepath = filepath + ".tmp"

    torch.save(checkpoint, temporary_filepath)
    os.replace(temporary_filepath, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"checkpoint 不存在：{filepath}"
        )

    checkpoint = torch.load(
        filepath,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )

    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )

    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    if restore_rng:
        restore_random_state(
            checkpoint.get("random_state")
        )

    epoch = checkpoint.get("epoch", 0)

    return {
        "start_epoch": epoch + 1,
        "global_step": checkpoint.get("global_step", 0),
        "best_dev_macro_f1": checkpoint.get(
            "best_dev_macro_f1",
            -1.0,
        ),
        "train_config": checkpoint.get(
            "train_config",
            {},
        ),
        "dev_metrics": checkpoint.get(
            "dev_metrics",
            {},
        ),
        "model_metadata": checkpoint.get(
            "model_metadata",
            {},
        ),
        "checkpoint_version": checkpoint.get(
            "checkpoint_version",
            0,
        ),
        "early_stopping_counter": checkpoint.get(
            "early_stopping_counter",
            0,
        ),
    }