import os
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from checkpoint import load_checkpoint, save_checkpoint
from dataset import build_dataloaders
from metrics import ClassificationMetrics
from model import BertTextClassifier

try:
    import swanlab
except ImportError:
    swanlab = None


class Trainer:
    def __init__(self, params):
        self.params = params

        random.seed(params.seed)
        torch.manual_seed(params.seed)

        device_name = params.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device_name)
        print("训练设备：", self.device)

        self.use_swanlab = (params.swanlab_enabled and swanlab is not None)
        self.swanlab_started = False

        self.start_epoch = 1
        self.global_step = 0
        self.best_dev_macro_f1 = -1.0
        self.epochs_without_improvement = 0

    def _start_swanlab(self):
        if self.params.swanlab_enabled and swanlab is None:
            print("未安装 swanlab，本次训练不记录在线日志")
            print("安装命令：pip install swanlab")
            return

        if not self.use_swanlab:
            return

        swanlab.init(
            project=self.params.swanlab_project,
            experiment_name=(self.params.swanlab_experiment_name),
            config=self.params.to_dict(),
        )
        self.swanlab_started = True

    def _prepare_training(self):
        params = self.params
        self.tokenizer = BertTokenizer.from_pretrained(params.model_path)
        self.train_loader,self.dev_loader,self.test_loader = build_dataloaders(tokenizer=self.tokenizer,params=params)

        self.model = BertTextClassifier(
            model_path=params.model_path,
            num_labels=params.num_labels,
            dropout=params.dropout,
        ).to(self.device)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )

        self.total_training_steps = (len(self.train_loader) * params.epochs)
        self.warmup_steps = int(self.total_training_steps * params.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps,
        )

        self.experiment_dir = os.path.join(params.save_dir,params.swanlab_experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.experiment_dir,"best_model.pt")
        self.last_checkpoint_path = os.path.join( self.experiment_dir,"last_checkpoint.pt")

        tokenizer_dir = os.path.join(self.experiment_dir,"tokenizer")
        self.tokenizer.save_pretrained(tokenizer_dir)

        print("总训练步数：", self.total_training_steps)
        print("Warmup 步数：", self.warmup_steps)

    def _resume_if_needed(self):
        if not self.params.resume_training:
            return

        resume_info = load_checkpoint(
            filepath=self.last_checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
            restore_rng=True,
        )

        self.start_epoch = resume_info["start_epoch"]
        self.global_step = resume_info["global_step"]
        self.best_dev_macro_f1 = resume_info[
            "best_dev_macro_f1"
        ]
        self.epochs_without_improvement = resume_info.get("early_stopping_counter",0)
        print(f"从第 {self.start_epoch} 个 epoch 继续训练")

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        total_batches = len(self.train_loader)

        for batch_index, batch in enumerate(self.train_loader,start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(input_ids=input_ids,attention_mask=attention_mask)
            loss = self.loss_function(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.params.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1

            should_log = ( batch_index % self.params.log_interval == 0 or batch_index == total_batches)
            if should_log:
                average_loss = total_loss / total_samples
                current_lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Batch {batch_index}/{total_batches} | "
                    f"Loss {average_loss:.4f} | "
                    f"LR {current_lr:.2e}"
                )

                if self.use_swanlab:
                    swanlab.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/average_loss": average_loss,
                            "train/learning_rate": current_lr,
                        },
                        step=self.global_step,
                    )

        if total_samples == 0:
            raise ValueError("训练数据集为空")

        return total_loss / total_samples

    def evaluate(self, data_loader):
        self.model.eval()
        metrics = ClassificationMetrics(num_classes=self.params.num_labels,class_names=self.params.label_names)
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids=input_ids,attention_mask=attention_mask)
                loss = self.loss_function(logits, labels)
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

    def _save_training_state(self,filepath,epoch,dev_result):
        checkpoint_metrics = {
            key: value
            for key, value in dev_result.items()
            if key != "report"
        }

        save_checkpoint(
            filepath=filepath,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            global_step=self.global_step,
            best_dev_macro_f1=self.best_dev_macro_f1,
            train_config=self.params.to_dict(),
            dev_metrics=checkpoint_metrics,
            early_stopping_counter=(
                self.epochs_without_improvement
            ),
        )

    def test(self):
        print()
        print("=" * 60)
        print("加载最佳模型并评价测试集")
        print("=" * 60)

        load_checkpoint(
            filepath=self.best_checkpoint_path,
            model=self.model,
            device=str(self.device),
            restore_rng=False,
        )

        test_result = self.evaluate(self.test_loader)

        print(f"Test Loss：{test_result['loss']:.4f}")
        print(f"Test Accuracy:{test_result['accuracy']:.4f}")
        print(f"Test Macro-F1：{test_result['macro_f1']:.4f}")
        print()
        print("测试集分类报告：")
        print(test_result["report"])

        if self.use_swanlab:
            swanlab.log(
                {
                    "test/loss": test_result["loss"],
                    "test/accuracy": test_result["accuracy"],
                    "test/macro_f1": test_result["macro_f1"],
                },
                step=self.global_step,
            )

        return test_result

    def fit(self):
        try:
            self._start_swanlab()
            self._prepare_training()
            self._resume_if_needed()

            for epoch in range(self.start_epoch,self.params.epochs + 1):
                print()
                print("=" * 60)
                print(f"Epoch {epoch}/{self.params.epochs}")
                print("=" * 60)

                train_loss = self.train_one_epoch()
                dev_result = self.evaluate(self.dev_loader)

                print(f"Train Loss：{train_loss:.4f}")
                print(f"Dev Loss：{dev_result['loss']:.4f}")
                print(f"Dev Accuracy：{dev_result['accuracy']:.4f}")
                print(f"Dev Macro-F1：{dev_result['macro_f1']:.4f}")

                if self.use_swanlab:
                    swanlab.log(
                        {
                            "train/epoch_loss": train_loss,
                            "dev/loss": dev_result["loss"],
                            "dev/accuracy": dev_result["accuracy"],
                            "dev/macro_precision": dev_result[
                                "macro_precision"
                            ],
                            "dev/macro_recall": dev_result[
                                "macro_recall"
                            ],
                            "dev/macro_f1": dev_result["macro_f1"],
                            "epoch": epoch,
                        },
                        step=self.global_step,
                    )

                current_macro_f1 = dev_result["macro_f1"]
                is_best = current_macro_f1 > (self.best_dev_macro_f1+ self.params.early_stopping_min_delta)

                if is_best:
                    self.best_dev_macro_f1 = current_macro_f1
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                self._save_training_state(filepath=self.last_checkpoint_path,epoch=epoch,dev_result=dev_result)
                print("已保存最近训练状态：",self.last_checkpoint_path)

                if is_best:
                    self._save_training_state(filepath=self.best_checkpoint_path,epoch=epoch,dev_result=dev_result)
                    print(f"已保存最佳模型：dev Macro-F1={self.best_dev_macro_f1:.4f}")
                patience = self.params.early_stopping_patience
                if patience > 0:
                    print(f"早停计数：{self.epochs_without_improvement}/{patience}")
                    if self.epochs_without_improvement >= patience:
                        print(f"验证集 Macro-F1 连续{patience}轮没有提升，提前结束训练")
                        break
            return self.test()
        finally:
            if self.swanlab_started:
                swanlab.finish()
