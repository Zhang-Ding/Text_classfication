from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import yaml


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class TrainingConfig:
    config_path: str
    experiment_name: str

    data_dir: str
    train_file: str
    dev_file: str
    test_file: str
    model_path: str
    save_dir: str

    label_codes: Tuple[int, ...]
    label_names: Tuple[str, ...]

    max_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    dropout: float
    max_grad_norm: float

    seed: int
    num_workers: int
    log_interval: int
    device: str
    resume_training: bool

    early_stopping_patience: int
    early_stopping_min_delta: float

    swanlab_enabled: bool
    swanlab_project: str

    @classmethod
    def from_yaml(cls, filepath: str) -> "TrainingConfig":
        config_path = Path(filepath).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_path = config_path.resolve()

        if not config_path.is_file():
            raise FileNotFoundError(
                f"配置文件不存在：{config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        if not isinstance(config_data, dict):
            raise ValueError("YAML 配置内容必须是键值结构")

        section_names = (
            "experiment",
            "data",
            "model",
            "training",
            "early_stopping",
            "output",
            "swanlab",
        )
        for section_name in section_names:
            if not isinstance(config_data.get(section_name), dict):
                raise ValueError(
                    f"配置文件缺少 {section_name} 配置段"
                )

        labels = config_data.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError("labels 必须是非空列表")

        try:
            label_codes = tuple(
                int(label["code"])
                for label in labels
            )
            label_names = tuple(
                str(label["name"])
                for label in labels
            )

            experiment = config_data["experiment"]
            data = config_data["data"]
            model = config_data["model"]
            training = config_data["training"]
            early_stopping = config_data["early_stopping"]
            output = config_data["output"]
            swanlab_config = config_data["swanlab"]
            data_dir = cls._resolve_path(data["data_dir"])

            params = cls(
                config_path=str(config_path),
                experiment_name=str(experiment["name"]),
                data_dir=data_dir,
                train_file=cls._resolve_path(
                    Path(data_dir) / str(data["train_file"])
                ),
                dev_file=cls._resolve_path(
                    Path(data_dir) / str(data["dev_file"])
                ),
                test_file=cls._resolve_path(
                    Path(data_dir) / str(data["test_file"])
                ),
                model_path=cls._resolve_path(model["model_path"]),
                save_dir=cls._resolve_path(output["save_dir"]),
                label_codes=label_codes,
                label_names=label_names,
                max_length=int(model["max_length"]),
                batch_size=int(training["batch_size"]),
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                warmup_ratio=float(training["warmup_ratio"]),
                weight_decay=float(training["weight_decay"]),
                dropout=float(model["dropout"]),
                max_grad_norm=float(training["max_grad_norm"]),
                seed=int(training["seed"]),
                num_workers=int(training["num_workers"]),
                log_interval=int(training["log_interval"]),
                device=str(training["device"]),
                resume_training=bool(
                    training["resume_training"]
                ),
                early_stopping_patience=int(
                    early_stopping["patience"]
                ),
                early_stopping_min_delta=float(
                    early_stopping["min_delta"]
                ),
                swanlab_enabled=bool(swanlab_config["enabled"]),
                swanlab_project=str(swanlab_config["project"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"配置文件参数格式错误：{error}"
            ) from error

        params.validate()
        return params

    @staticmethod
    def _resolve_path(filepath: str) -> str:
        if not str(filepath).strip():
            raise ValueError("文件路径不能为空")

        path = Path(filepath).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path

        return str(path.resolve())

    @property
    def label2id(self) -> Dict[int, int]:
        return {
            label_code: label_id
            for label_id, label_code in enumerate(self.label_codes)
        }

    @property
    def id2label(self) -> Dict[int, int]:
        return {
            label_id: label_code
            for label_code, label_id in self.label2id.items()
        }

    @property
    def num_labels(self) -> int:
        return len(self.label_codes)

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment.name 不能为空")

        if len(self.label_codes) != len(self.label_names):
            raise ValueError("标签编号和标签名称数量不一致")

        if len(set(self.label_codes)) != len(self.label_codes):
            raise ValueError("标签编号不能重复")

        if len(set(self.label_names)) != len(self.label_names):
            raise ValueError("标签名称不能重复")

        if self.epochs <= 0:
            raise ValueError("training.epochs 必须大于 0")
        if self.batch_size <= 0:
            raise ValueError("training.batch_size 必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate 必须大于 0")
        if self.max_length <= 0:
            raise ValueError("model.max_length 必须大于 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout 必须在 [0, 1) 范围内")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(
                "training.warmup_ratio 必须在 [0, 1] 范围内"
            )
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay 不能小于 0")
        if self.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm 必须大于 0")
        if self.seed < 0:
            raise ValueError("training.seed 不能小于 0")
        if self.num_workers < 0:
            raise ValueError("training.num_workers 不能小于 0")
        if self.log_interval <= 0:
            raise ValueError("training.log_interval 必须大于 0")
        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping.patience 不能小于 0"
            )
        if self.early_stopping_min_delta < 0:
            raise ValueError(
                "early_stopping.min_delta 不能小于 0"
            )

    def to_dict(self) -> dict:
        config_data = asdict(self)
        config_data["label_codes"] = list(self.label_codes)
        config_data["label_names"] = list(self.label_names)
        config_data["num_labels"] = self.num_labels
        return config_data
