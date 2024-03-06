from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


import torch
from torch.utils.data import Dataset


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1).long()) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"x": pixel_values, "labels": labels}


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["x"])
        target = inputs["labels"]
        loss = F.cross_entropy(outputs, target)
        return (loss, outputs) if return_outputs else loss


def train():
    train_set, model, optimizer = load_train_objs()

    training_args = TrainingArguments(
        "basic-trainer", per_device_train_batch_size=64, num_train_epochs=1000, remove_unused_columns=False
    )

    trainer = MyTrainer(
        model,
        training_args,
        train_dataset=train_set,
        data_collator=collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    train()

# accelerate launch accelerate.py
