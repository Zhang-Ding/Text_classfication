import argparse

from config import TrainingConfig


def parse_training_config() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="使用 YAML 配置训练 BERT 文本分类模型"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML 配置文件路径",
    )

    args = parser.parse_args()
    return TrainingConfig.from_yaml(args.config)
