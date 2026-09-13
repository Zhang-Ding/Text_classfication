import os
from dataclasses import asdict,dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR,"data")
MODEL_PATH = os.path.join(BASE_DIR,"bertmodel","bert-base-chinese")

TRAIN_FILE = os.path.join(DATA_DIR,"train_3k.txt")
DEV_FILE = os.path.join(DATA_DIR,"dev_1k.txt")
TEST_FILE = os.path.join(DATA_DIR,"test_1k.txt")

SAVE_DIR = os.path.join(BASE_DIR,"checkpoints")
os.makedirs(SAVE_DIR,exist_ok = True)
#标签映射
LABEL2ID = {
    100: 0,
    101: 1,
    102: 2,
    103: 3,
    104: 4,
    106: 5,
    107: 6,
    108: 7,
    109: 8,
    110: 9,
    112: 10,
    113: 11,
    114: 12,
    115: 13,
    116: 14,
}

LABEL_NAMES = [
    "news_story",
    "news_culture",
    "news_entertainment",
    "news_sports",
    "news_finance",
    "news_house",
    "news_car",
    "news_edu",
    "news_tech",
    "news_military",
    "news_travel",
    "news_world",
    "stock",
    "news_agriculture",
    "news_game",
]


ID2LABEL = {
    label_id : label_code
    for label_code,label_id in LABEL2ID.items()
}

NUM_LABELS = len(LABEL2ID)

#训练参数
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
SEED = 42

#swnlab
SWANLAB_PROJECT = "bert-text-classification"
SWANLAB_EXPERIMENT_NAME = (
    f"bert-lr{LEARNING_RATE}-"
    f"batch{BATCH_SIZE}-"
    f"epoch{EPOCHS}"
)

@dataclass
class TrainingConfig:
    train_file: str = TRAIN_FILE
    dev_file: str = DEV_FILE
    test_file: str = TEST_FILE

    model_path: str = MODEL_PATH
    save_dir: str = SAVE_DIR

    num_labels: int = NUM_LABELS
    label_names: tuple = tuple(LABEL_NAMES)

    max_length: int = MAX_LENGTH
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    warmup_ratio: float = WARMUP_RATIO
    weight_decay: float = WEIGHT_DECAY
    dropout: float = DROPOUT
    max_grad_norm: float = 1.0

    seed: int = SEED
    num_workers: int = 0
    log_interval: int = 20

    device: str = "auto"
    resume_training: bool = False

    swanlab_enabled: bool = True
    swanlab_project: str = SWANLAB_PROJECT
    swanlab_experiment_name: str = (
        SWANLAB_EXPERIMENT_NAME
    )

    def to_dict(self) -> dict:
        return asdict(self)

if __name__ == "__main__":
    print("项目根目录：", BASE_DIR)
    print("训练集：", TRAIN_FILE)
    print("验证集：", DEV_FILE)
    print("测试集：", TEST_FILE)
    print("本地模型：", MODEL_PATH)
    print("类别数量：", NUM_LABELS)

    print("训练集存在：", os.path.isfile(TRAIN_FILE))
    print("验证集存在：", os.path.isfile(DEV_FILE))
    print("测试集存在：", os.path.isfile(TEST_FILE))
    print("模型目录存在：", os.path.isdir(MODEL_PATH))

