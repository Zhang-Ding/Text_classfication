import os
import random
import torch

# 当前代码使用的 checkpoint 文件格式版本
CHECKPOINT_VERSION = 1


def get_random_state():
    random_state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }

    if torch.cuda.is_available():
        random_state["cuda"] = torch.cuda.get_rng_state_all()

    return random_state


def restore_random_state(random_state):
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


def get_model_metadata(model):
    metadata = {"model_class": model.__class__.__name__}

    if hasattr(model, "bert"):
        bert = model.bert

        if hasattr(bert, "config"):
            metadata["bert_config"] = bert.config.to_dict()

    return metadata


def save_checkpoint(filepath,model,optimizer,scheduler,epoch,global_step,best_dev_macro_f1,train_config,dev_metrics,early_stopping_counter) -> None:
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


def load_checkpoint(filepath,model,optimizer = None,scheduler = None,device = "cpu",restore_rng = True):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"checkpoint 不存在：{filepath}"
        )

    checkpoint = torch.load(
        filepath,
        map_location=device,
        weights_only=False,
    )

    # 加载模型状态前检查文件格式版本，避免读取不兼容的新版本 checkpoint
    # 旧 checkpoint 没有该字段时，按版本 0 处理
    checkpoint_version = checkpoint.get(
    "checkpoint_version",
    0,
    )

    if checkpoint_version > CHECKPOINT_VERSION:
        raise RuntimeError(
            "当前代码无法读取该 checkpoint："
            f"文件版本为 {checkpoint_version}，"
            f"代码支持的最高版本为 {CHECKPOINT_VERSION}"
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