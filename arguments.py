import argparse
from dataclasses import replace

from config import TrainingConfig


def parse_training_config() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="训练 BERT 文本分类模型"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每个 batch 的样本数量",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学习率",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="文本最大 token 数",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 工作进程数",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="auto、cpu、cuda、cuda:0 等",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="SwanLab 实验名称",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 last_checkpoint.pt 继续训练",
    )

    parser.add_argument(
        "--disable-swanlab",
        action="store_true",
        help="关闭 SwanLab 实验日志",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="早停等待轮数，设为0表示关闭",
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Macrof1被视为提升所需的最小变化",
    )

    args = parser.parse_args()

    params = TrainingConfig()
    overrides = {}

    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    if args.max_length is not None:
        overrides["max_length"] = args.max_length

    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers

    if args.device is not None:
        overrides["device"] = args.device

    if args.experiment_name is not None:
        overrides["swanlab_experiment_name"] = (
            args.experiment_name
        )

    if args.resume:
        overrides["resume_training"] = True

    if args.disable_swanlab:
        overrides["swanlab_enabled"] = False

    if args.patience is not None:
        overrides["early_stopping_patience"] = args.patience

    if args.min_delta is not None:
        overrides["early_stopping_min_delta"] = args.min_delta

    params = replace(params, **overrides)

    if args.experiment_name is None:
        generated_name = (
            f"bert-lr{params.learning_rate}-"
            f"batch{params.batch_size}-"
            f"epoch{params.epochs}"
        )

        params = replace(
            params,
            swanlab_experiment_name=generated_name,
        )

    return params


if __name__ == "__main__":
    params = parse_training_config()

    print("epochs：", params.epochs)
    print("batch_size：", params.batch_size)
    print("learning_rate：", params.learning_rate)
    print("max_length：", params.max_length)
    print("num_workers：", params.num_workers)
    print("device：", params.device)
    print("resume_training：", params.resume_training)
    print("swanlab_enabled：", params.swanlab_enabled)
    print("experiment_name：",params.swanlab_experiment_name)