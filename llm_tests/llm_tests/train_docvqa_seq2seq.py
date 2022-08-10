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
from llm_tests.budget_metrics_s2s import compute_anls_metric
from transformers import AutoProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.data.data_collator import default_data_collator
from transformers import TrainingArguments, Trainer, EvalPrediction
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import load_metric
from llm_tests.modeling_llm3dec import LayoutLMv3Seq2SeqModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from llm_tests.s2s_trainer import LLMBartSeq2SeqTrainer

from llm_tests.budget_metrics_shared import normalized_levenshtein

from tqdm.auto import tqdm

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
    fp16: Optional[bool] = False,
    gradient_checkpointing: Optional[bool] = False,
    log_wandb: Optional[bool] = False,
    project_id: Optional[str] = '"llm3-docvqa',
):
    # load the model
    print(f"loading model: {model_path}")
    model = LayoutLMv3Seq2SeqModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    decoder_tokenizer = AutoTokenizer.from_pretrained(model.config.decoder._name_or_path)
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

    steps_per_epoch = len(train_data) // batch
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./train_output_{run_name}",
        overwrite_output_dir=True,
        max_steps=steps,
        learning_rate=lr,
        # warmup_steps=int(steps * warmup_ratio),
        # warmup_ratio=warmup_ratio,
        save_steps=save_every,
        # evaluation_strategy = "epoch",
        evaluation_strategy="steps",
        eval_steps=steps_per_epoch // 2,
        per_device_train_batch_size=inst_batch // n_devices,
        per_device_eval_batch_size=inst_batch // n_devices,
        gradient_accumulation_steps=batch // inst_batch,
        run_name=run_name,
        report_to = "wandb" if log_wandb else None,
        predict_with_generate = True,
        logging_steps = steps_per_epoch // 10 + 1,
    )
    print('training_args:', training_args)

    if fp16:
        training_args.fp16 = True
    
    if gradient_checkpointing:
        training_args.gradient_checkpointing = True

    optimizer = AdamW(model.parameters(), lr=lr)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(steps * warmup_ratio),
    #     num_training_steps=steps
    # )
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(steps * warmup_ratio),
        num_training_steps=steps,
        num_cycles=10,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # print(f"predictions: {predictions}")
        # print(f"labels: {labels}")

        # print("num preds:" , len(predictions))
        # print("num preds[0]:", len(predictions[0]))
        # print("what's in preds:", predictions)
        # print("num preds[0][0]:", len(predictions[0][0]))

        # print("num labels:", len(labels))
        # print("num labels[0]:", len(labels[0]))

        # decode the predictions
        decoded_preds = decoder_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # replace label -100 with padding token
        labels = np.where(labels != -100, labels, decoder_tokenizer.pad_token_id)
        # decode the labels
        decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # map decoded labels to acceptable_answers
        acceptable_answers_list = [[x] for x in decoded_labels]

        # compute the metrics
        result = compute_anls_metric(decoded_preds, acceptable_answers_list)

        # print(f"anls result: {result}")

        return result

    # set up the trainer
    trainer = LLMBartSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=processor,
        data_collator=default_data_collator,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=compute_metrics,
    )
    trainer.set_generation_kwargs(do_sample=True, top_p=0.9, temperature=0.2)

    # evaluate the model
    # trainer.evaluate()

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
