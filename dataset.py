from typing import List,Tuple
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,DataCollatorWithPadding
from config import LABEL2ID,TrainingConfig

#加载原始数据
def load_raw_data(filepath: str) -> List[Tuple[str, int]]:
    samples = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                skipped += 1
                continue
            parts = line.split("_!_", maxsplit=4)
            if len(parts) < 4:
                print(f"第 {line_number} 行字段不完整，已跳过")
                skipped += 1
                continue
            label_text = parts[1].strip()
            title = parts[3].strip()
            try:
                label_code = int(label_text)
            except ValueError:
                print(
                    f"第 {line_number} 行标签不是整数："
                    f"{label_text!r}，已跳过"
                )
                skipped += 1
                continue
            if label_code not in LABEL2ID:
                print(
                    f"第 {line_number} 行出现未知标签："
                    f"{label_code}，已跳过"
                )
                skipped += 1
                continue

            if not title:
                print(f"第 {line_number} 行标题为空，已跳过")
                skipped += 1
                continue

            label_id = LABEL2ID[label_code]
            samples.append((title, label_id))

    print(f"数据文件：{filepath}")
    print(f"有效样本：{len(samples)}")
    print(f"跳过样本：{skipped}")

    if not samples:
        raise ValueError(f"没有从文件中读取到有效样本：{filepath}")

    return samples

#dataset类
class TextClassificationDataset(Dataset):
    def __init__(self,samples:List[Tuple[str,int]],tokenizer:BertTokenizer,max_length:int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self)->int:
        return len(self.samples)

    def __getitem__(self,index:int)->dict:
        text, label_id = self.samples[index]

        encoding = self.tokenizer(
            text,
            max_length = self.max_length,
            truncation = True,
            padding = False,
        )

        return{
            "input_ids" : encoding["input_ids"],
            "attention_mask" : encoding["attention_mask"],
            "labels" : label_id,
        }

def build_dataloader(filepath:str,tokenizer:BertTokenizer,max_length:int,batch_size:int,shuffle:bool,num_workers:int)->DataLoader:
    samples = load_raw_data(filepath)

    dataset = TextClassificationDataset(
        samples = samples,
        tokenizer = tokenizer,
        max_length = max_length,
    )

    collator = DataCollatorWithPadding(
        tokenizer = tokenizer,
        return_tensors = "pt",
    )

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        collate_fn = collator,
        num_workers = num_workers,
        )

    return dataloader

def build_dataloaders(tokenizer: BertTokenizer,params:TrainingConfig)->Tuple[DataLoader,DataLoader,DataLoader]:
    common_arguments = {
        "tokenizer": tokenizer,
        "batch_size": params.batch_size,
        "max_length": params.max_length,
        "num_workers": params.num_workers
    }
    train_loader = build_dataloader(
        filepath = params.train_file,
        shuffle = True,
        **common_arguments,
    )

    dev_loader = build_dataloader(
        filepath = params.dev_file,
        shuffle = False,
        **common_arguments,
    )

    test_loader = build_dataloader(
        filepath=params.test_file,
        shuffle=False,
        **common_arguments,
    )

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    params = TrainingConfig()

    tokenizer = BertTokenizer.from_pretrained(
        params.model_path
    )

    train_loader, dev_loader, test_loader = (
        build_dataloaders(
            tokenizer=tokenizer,
            params=params,
        )
    )

    print("训练集 batch 数：", len(train_loader))
    print("验证集 batch 数：", len(dev_loader))
    print("测试集 batch 数：", len(test_loader))

    first_batch = next(iter(train_loader))

    print(
        "input_ids 形状：",
        first_batch["input_ids"].shape,
    )
    print(
        "attention_mask 形状：",
        first_batch["attention_mask"].shape,
    )
    print(
        "labels 形状：",
        first_batch["labels"].shape,
    )

    first_label = first_batch["labels"][0].item()

    print("第一条标签编号：", first_label)
    print(
        "第一条标签名称：",
        params.label_names[first_label],
    )

    first_text = tokenizer.decode(
        first_batch["input_ids"][0],
        skip_special_tokens=True,
    )

    print("第一条还原文本：", first_text)