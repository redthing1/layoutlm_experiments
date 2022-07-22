import typer
import os
from typing import Optional, Dict, Tuple
import collections
import json
import logging

import torch
import pandas as pd
import numpy as np

from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer
from transformers.data.data_collator import default_data_collator
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_metric

from tqdm.auto import tqdm
from llm_tests.qa_trainer import QuestionAnsweringTrainer

logger = logging.getLogger(__name__)

def cli(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    run_name: str,
    # device: Optional[str] = 'cpu',
    checkpoint: str = None,
    batch: Optional[int] = 128,
    inst_batch: Optional[int] = 16,
    lr: Optional[float] = 1e-5,
    steps: Optional[int] = 100000,
    warmup_ratio: Optional[float] = 0.1,
    save_every: Optional[int] = 1000,
    log_wandb: Optional[bool] = False,
    project_id: Optional[str] = '"llm3-docvqa',
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

    # wandb setup if enabled
    if log_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=project_id,
            name=run_name
        )

    n_devices = 1
    # check if gpu available
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        print(f"using {n_devices} gpu")

    training_args = TrainingArguments(
        output_dir=f"./train_output_{run_name}",
        overwrite_output_dir=True,
        max_steps=steps,
        learning_rate=lr,
        # warmup_steps=int(steps * warmup_ratio),
        warmup_ratio=warmup_ratio,
        save_steps=save_every,
        evaluation_strategy = "epoch",
        per_device_train_batch_size=inst_batch // n_devices,
        per_device_eval_batch_size=inst_batch // n_devices,
        gradient_accumulation_steps=batch // inst_batch,
        run_name=run_name,
        report_to = "wandb" if log_wandb else None,
    )
    print('training_args:', training_args)

    # set up the trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=processor,
        data_collator=default_data_collator,
    )

    # train the model
    print('starting training')

    # if checkpoint is specified, load it
    if checkpoint is not None:
        print(f"loading checkpoint: {checkpoint}")
        train_results = trainer.train(checkpoint)
    else:
        train_results = trainer.train()

    # save out the model
    print('saving trained model')
    model.save_model(training_args.output_dir + '/' + run_name + '_train_finish')
    print('model saved')

    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    trainer.save_state()

    def log_eval():
        # evaluate the model
        print('evaluating model')
        eval_results = trainer.evaluate()
        print('evaluation results:', eval_results)

    log_eval()

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
