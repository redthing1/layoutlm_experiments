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

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    # logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    return all_predictions

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

    metric = load_metric("squad")

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            # version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=20,
            max_answer_length=512,
            null_score_diff_threshold=0,
            output_dir=training_args.output_dir,
            # log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(p: EvalPrediction):
        print(f'computing metrics for: {p}')
        metrics_results = metric.compute(predictions=p.predictions, references=p.label_ids)
        print(f"metrics_results: {metrics_results}")
        return metrics_results

    # set up the trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=processor,
        data_collator=default_data_collator,
        # device=torch.device(device),
        # compute_metrics=compute_metrics,
    )

    def log_eval():
        # evaluate the model
        print('evaluating model')
        eval_results = trainer.evaluate()
        print('evaluation results:', eval_results)
    
    log_eval()

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

    log_eval()

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
