from typing import Iterable,Optional,Sequence

class ClassificationMetrics:
    def __init__(self,num_classes:int,class_names:Optional[Sequence[str]]=None):
        if num_classes<=0:
            raise ValueError("num_classes必须大于0")
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = [
                str(class_id)
                for class_id in range (num_classes)
            ]
        else:
            if len(class_names)!= num_classes:
                raise ValueError("class_names 的长度必须等于 num_classes")
            self.class_names = list(class_names)
        self.reset()

    #清空历史结果
    def reset(self)->None:
        self.confusion_matrix =[
            [0 for _ in range(self.num_classes)]
            for _ in range(self.num_classes)
        ]
    
    #累计一个batch,接收一批真实标签和预测标签，把结果累计到混淆矩阵中
    def update(self,labels:Iterable[int],predictions:Iterable[int])->None:
        labels = list(labels)
        predictions = list(predictions)

        if len(labels) != len(predictions):
            raise ValueError("labels 和 predictions 的长度必须一致")

        for label,prediction in zip(labels,predictions):
            label = int(label)
            prediction = int(prediction)

            if not 0 <= label < self.num_classes:
                raise ValueError(
                    f"真实标签超出范围：{label}"
                )

            if not 0 <= prediction < self.num_classes:
                raise ValueError(
                    f"预测标签超出范围：{prediction}"
                )

            self.confusion_matrix[label][prediction] += 1

    #防止除以0，复用
    @staticmethod
    def safe_divide(numerator:float,denominator:float)->float:
        if denominator==0:
            return 0.0
        return numerator/denominator

    def compute(self)->dict:
        total_samples = sum(sum(row) for row in self.confusion_matrix)

        correct_samples = sum(
            self.confusion_matrix[class_id][class_id]
            for class_id in range (self.num_classes)
        )

        accuracy = self.safe_divide(correct_samples,total_samples)

        per_class={}

        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0

        for class_id in range(self.num_classes):
            true_positive = self.confusion_matrix[class_id][class_id]

            #预测为当前类别的数量
            predicted_positive = sum(
                self.confusion_matrix[row_id][class_id]
                for row_id in range(self.num_classes)
            )

            #计算当前类别所在行的总和
            actual_positive = sum(self.confusion_matrix[class_id])

            false_positive = (predicted_positive-true_positive)
            false_negative = (actual_positive-true_positive)

            precision = self.safe_divide(true_positive,true_positive+false_positive)
            recall = self.safe_divide(true_positive,true_positive+false_negative)
            f1 = self.safe_divide(2*precision*recall,precision+recall)

            #获取类别名称
            class_name = self.class_names[class_id]
            #保存当前类别结果
            per_class[class_name] = {
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actual_positive,#该类别真实样本数量
            }

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1

        #计算Macro平均值
        macro_precision = ( precision_sum / self.num_classes)
        macro_recall = (recall_sum / self.num_classes)
        macro_f1 = ( f1_sum / self.num_classes)

        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class": per_class,          
            "confusion_matrix": [
                row.copy()#复制混淆矩阵，这里返回矩阵副本，外部修改返回结果不影响内部状态。
                for row in self.confusion_matrix
            ],
            "total_samples": total_samples,
        }

    def report(self, digits: int = 4) -> str:
        result = self.compute()

        lines = []

        header = (
            f"{'class':<24}"
            f"{'precision':>12}"
            f"{'recall':>12}"
            f"{'f1-score':>12}"
            f"{'support':>10}"
        )
        lines.append(header)

        for class_name, values in result["per_class"].items():
            line = (
                f"{class_name:<24}"
                f"{values['precision']:>12.{digits}f}"
                f"{values['recall']:>12.{digits}f}"
                f"{values['f1']:>12.{digits}f}"
                f"{values['support']:>10d}"
            )
            lines.append(line)

        lines.append("")

        accuracy_line = (
            f"{'accuracy':<24}"
            f"{'':>12}"
            f"{'':>12}"
            f"{result['accuracy']:>12.{digits}f}"
            f"{result['total_samples']:>10d}"
        )
        lines.append(accuracy_line)

        macro_line = (
            f"{'macro avg':<24}"
            f"{result['macro_precision']:>12.{digits}f}"
            f"{result['macro_recall']:>12.{digits}f}"
            f"{result['macro_f1']:>12.{digits}f}"
            f"{result['total_samples']:>10d}"
        )
        lines.append(macro_line)

        return "\n".join(lines)

if __name__ == "__main__":
    labels = [
        0, 0,
        1, 1,
        2, 2,
    ]

    predictions = [
        0, 1,
        1, 1,
        0, 2,
    ]

    metrics = ClassificationMetrics(
        num_classes=3,
        class_names=[
            "class_0",
            "class_1",
            "class_2",
        ],
    )

    metrics.update(
        labels=labels,
        predictions=predictions,
    )

    result = metrics.compute()

    print("混淆矩阵：")
    for row in result["confusion_matrix"]:
        print(row)

    print()
    print(metrics.report())





    

    



