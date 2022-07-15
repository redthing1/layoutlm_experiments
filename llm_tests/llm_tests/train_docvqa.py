import typer
import os
from typing import Optional

import torch
import json
import pandas as pd

from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer
from transformers.data.data_collator import default_data_collator
from transformers import TrainingArguments, Trainer

def cli(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
):
    # load the model
    print(f"loading model: {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    print(f"model loaded: {model_path}")

    # load the data
    print(f"loading train data: {train_data_path}")
    train_data = load_dataset(train_data_path)
    print(f"loading val data: {val_data_path}")
    val_data = load_dataset(val_data_path)

    print(f"train data: {train_data}")
    print(f"val data: {val_data}")

    # set up the data
    train_data.set_format("torch")
    val_data.set_format("torch")

    # training_args = TrainingArguments(

    # )


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
