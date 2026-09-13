import torch
import torch.nn as nn
from transformers import BertModel

import config

class BertTextClassifier(nn.Module):
    def __init__(self,model_path:str,num_labels:int,dropout:float):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Linear(
            in_features = hidden_size,
            out_features = num_labels,
        )


    def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor)->torch.Tensor:
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True,
        )

        cls_vector = outputs.last_hidden_state[:,0,:]
        cls_vector = self.dropout(cls_vector)

        logits = self.classifier(cls_vector)

        return logits
    
if __name__ == "__main__":
    from transformers import BertTokenizer

    from dataset import build_dataloaders

    tokenizer = BertTokenizer.from_pretrained(
        config.MODEL_PATH
    )

    train_loader, _, _ = build_dataloaders(tokenizer)

    batch = next(iter(train_loader))

    model = BertTextClassifier(
        model_path=config.MODEL_PATH,
        num_labels=config.NUM_LABELS,
        dropout=config.DROPOUT,
    )

    model.eval()

    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    predictions = logits.argmax(dim=-1)

    print("input_ids 形状：", batch["input_ids"].shape)
    print("BERT 输出类别数：", logits.shape)
    print("预测标签形状：", predictions.shape)

    first_prediction = predictions[0].item()

    print("第一条预测编号：", first_prediction)
    print(
        "第一条预测类别：",
        config.LABEL_NAMES[first_prediction],
    )