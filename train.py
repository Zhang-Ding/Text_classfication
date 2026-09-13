import os
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer,get_linear_schedule_with_warmup

from arguments import parse_training_config
from checkpoint import load_checkpoint, save_checkpoint
from config import TrainingConfig
from dataset import build_dataloaders
from metrics import ClassificationMetrics
from model import BertTextClassifier

try:
    import swanlab
except ImportError:
    swanlab = None

def evaluate(model,data_loader,loss_function,device,params):
    model.eval()

    metrics = ClassificationMetrics(num_classes=params.num_labels,class_names=params.label_names)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids,attention_mask=attention_mask)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=-1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            metrics.update(
                labels=labels.cpu().tolist(),
                predictions=predictions.cpu().tolist(),
            )

    if total_samples == 0:
        raise ValueError("评价数据集为空")

    result = metrics.compute()
    result["loss"] = total_loss / total_samples
    result["report"] = metrics.report()

    return result


def train(params: TrainingConfig):
    random.seed(params.seed)
    torch.manual_seed(params.seed)

    device_name = (
        params.device
        if params.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )
    device = torch.device(device_name)
    print("训练设备：", device)

    os.makedirs(params.save_dir, exist_ok=True)
    use_swanlab = (params.swanlab_enabled and swanlab is not None)

    if params.swanlab_enabled and swanlab is None:
        print( "未安装 swanlab，本次训练不记录在线日志")
        print("安装命令：pip install swanlab")

    if use_swanlab:
        swanlab.init(
            project=params.swanlab_project,
            experiment_name=(
                params.swanlab_experiment_name
            ),
            config=params.to_dict(),
        )

    tokenizer = BertTokenizer.from_pretrained(params.model_path)
    train_loader, dev_loader, test_loader = (
        build_dataloaders(
            tokenizer=tokenizer,
            params=params,
        )
    )

    model = BertTextClassifier(
        model_path=params.model_path,
        num_labels=params.num_labels,
        dropout=params.dropout,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    total_training_steps = (len(train_loader) * params.epochs)
    warmup_steps = int(total_training_steps * params.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    #每组实验放进独立目录
    experiment_dir = os.path.join(params.save_dir,params.swanlab_experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(experiment_dir,"best_model.pt")

    last_checkpoint_path = os.path.join(experiment_dir,"last_checkpoint.pt")

    tokenizer.save_pretrained(os.path.join(experiment_dir, "tokenizer"))

    start_epoch = 1
    global_step = 0
    best_dev_macro_f1 = -1.0
    epochs_without_improvement = 0

    if params.resume_training:
        resume_info = load_checkpoint(
            filepath=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            restore_rng=True,
        )

        start_epoch = resume_info["start_epoch"]
        global_step = resume_info["global_step"]
        best_dev_macro_f1 = resume_info["best_dev_macro_f1"]

        epochs_without_improvement = resume_info.get(
        "early_stopping_counter",
        0,
        )

        print( f"从第 {start_epoch} 个 epoch 继续训练")

    print("总训练步数：", total_training_steps)
    print("Warmup 步数：", warmup_steps)

    for epoch in range(start_epoch,params.epochs + 1):
        print()
        print("=" * 60)
        print(f"Epoch {epoch}/{params.epochs}")
        print("=" * 60)

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for batch_index, batch in enumerate(train_loader,start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_function(logits,labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=params.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            batch_size = labels.size(0)
            total_train_loss += (loss.item() * batch_size)
            total_train_samples += batch_size
            global_step += 1

            should_log = (batch_index % params.log_interval == 0 or batch_index == len(train_loader))

            if should_log:
                average_train_loss = (total_train_loss/ total_train_samples)
                current_lr = (scheduler.get_last_lr()[0])
                print(
                    f"Batch "
                    f"{batch_index}/"
                    f"{len(train_loader)} | "
                    f"Loss "
                    f"{average_train_loss:.4f} | "
                    f"LR {current_lr:.2e}"
                )

                if use_swanlab:
                    swanlab.log(
                        {
                            "train/batch_loss": (loss.item()),
                            "train/average_loss": (average_train_loss),
                            "train/learning_rate": (current_lr),
                        },
                        step=global_step,
                    )

        train_loss = (total_train_loss/total_train_samples)

        dev_result = evaluate(
            model=model,
            data_loader=dev_loader,
            loss_function=loss_function,
            device=device,
            params=params,
        )

        print(f"Train Loss：{train_loss:.4f}")
        print(f"Dev Loss：{dev_result['loss']:.4f}")
        print("Dev Accuracy："f"{dev_result['accuracy']:.4f}")
        print("Dev Macro-F1：" f"{dev_result['macro_f1']:.4f}")

        if use_swanlab:
            swanlab.log(
                {
                    "train/epoch_loss": train_loss,
                    "dev/loss": dev_result["loss"],
                    "dev/accuracy": (dev_result["accuracy"]),
                    "dev/macro_precision": (dev_result["macro_precision"]),
                    "dev/macro_recall": (dev_result["macro_recall"]),
                    "dev/macro_f1": (dev_result["macro_f1"]),
                    "epoch": epoch,
                },
                step=global_step,
            )

        #最佳模型判断
        current_macro_f1 = dev_result["macro_f1"]
        is_best = (current_macro_f1> best_dev_macro_f1 + params.early_stopping_min_delta)

        if is_best:
            best_dev_macro_f1 = current_macro_f1
            epochs_without_improvement =0
        else:
            epochs_without_improvement +=1

        checkpoint_metrics = {
            key: value
            for key, value in dev_result.items()
            if key != "report"
        }

        save_checkpoint(
            filepath=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_dev_macro_f1=(best_dev_macro_f1),
            train_config=params.to_dict(),
            dev_metrics=checkpoint_metrics,
            early_stopping_counter=epochs_without_improvement,
        )

        print("已保存最近训练状态：",last_checkpoint_path)

        if is_best:
            save_checkpoint(
                filepath=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_dev_macro_f1=(best_dev_macro_f1),
                train_config=params.to_dict(),
                dev_metrics=checkpoint_metrics,
                early_stopping_counter=epochs_without_improvement,
            )

            print("已保存最佳模型："f"dev Macro-F1="f"{best_dev_macro_f1:.4f}")
        if params.early_stopping_patience >0:
            print(f"早停计数：{epochs_without_improvement}/{params.early_stopping_patience}")
            if(epochs_without_improvement>=params.early_stopping_patience):
                print(f"验证集 Macro-F1 连续{params.early_stopping_patience}轮没有提升，提前结束训练")
                break
    print()
    print("=" * 60)
    print("加载最佳模型并评价测试集")
    print("=" * 60)

    load_checkpoint(
        filepath=best_checkpoint_path,
        model=model,
        device=str(device),
        restore_rng=False,
    )

    test_result = evaluate(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
        params=params,
    )

    print(f"Test Loss：{test_result['loss']:.4f}")
    print("Test Accuracy："f"{test_result['accuracy']:.4f}")
    print("Test Macro-F1："f"{test_result['macro_f1']:.4f}")
    print()
    print("测试集分类报告：")
    print(test_result["report"])

    if use_swanlab:
        swanlab.log(
            {
                "test/loss": test_result["loss"],
                "test/accuracy": (test_result["accuracy"]),
                "test/macro_f1": (test_result["macro_f1"]),
            },
            step=global_step,
        )

        swanlab.finish()

    return test_result


def main():
    params = parse_training_config()
    train(params)


if __name__ == "__main__":
    main()