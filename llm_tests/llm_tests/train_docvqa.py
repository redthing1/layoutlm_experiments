import typer
import os
from typing import Optional

import torch
import json
import pandas as pd

from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer
from transformers.data.data_collator import default_data_collator
from transformers import TrainingArguments, Trainer

def cli(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    # device: Optional[str] = 'cpu',
    batch: Optional[int] = 128,
    device_batch: Optional[int] = 16,
    lr: Optional[float] = 1e-5,
    steps: Optional[int] = 100000,
    warmup_ratio: Optional[float] = 0.1,
    save_every: Optional[int] = 1000,
    eval_every: Optional[int] = 1000,
):
    # load the model
    print(f"loading model: {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    print(f"model loaded: {model_path}")

    # load the data
    print(f"loading train data: {train_data_path}")
    train_data = load_from_disk(train_data_path)
    print(f"loading val data: {val_data_path}")
    val_data = load_from_disk(val_data_path)

    print(f"train data: {train_data}")
    print(f"val data: {val_data}")

    # set up the data
    train_data.set_format("torch")
    val_data.set_format("torch")

    training_args = TrainingArguments(
        output_dir="./train_output",
        overwrite_output_dir=True,
        max_steps=steps // batch,
        learning_rate=lr,
        # warmup_steps=int(steps * warmup_ratio),
        warmup_ratio=warmup_ratio,
        eval_steps=save_every,
        save_steps=eval_every,
        per_device_train_batch_size=device_batch,
        per_device_eval_batch_size=device_batch,
        gradient_accumulation_steps=batch // device_batch,
    )
    print('training_args:', training_args)

    # set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=processor,
        data_collator=default_data_collator,
        # device=torch.device(device)
    )

    # train the model
    trainer.train()


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
