import typer
import os
import sys
import tempfile
import json
from typing import Optional
from types import SimpleNamespace
import importlib.util

import torch
import pandas as pd
from PIL import Image
import editdistance

from transformers import (
    AutoProcessor,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    LayoutLMv3FeatureExtractor,
)
from datasets import load_dataset, load_from_disk

def fuzzy(s1, s2, threshold=0.2):
    return (editdistance.eval(s1, s2) / ((len(s1) + len(s2)) / 2)) < threshold

def cli(
    model_path: str,
    val_data_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    print(f"loading model: {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)
    print(f"model loaded: {model_path}")

    print(f"loading val data: {val_data_path}")
    val_data = load_from_disk(val_data_path)
    print(f"val data loaded: {val_data_path}")

    val_data.set_format("torch")

    dataloader = torch.utils.data.DataLoader(val_data, batch_size=1)

    def check_preds(outputs, labels):
        answer_start_scores = outputs.start_logits.cpu().detach()
        answer_end_scores = outputs.end_logits.cpu().detach()

        answer_start_probs = answer_start_scores.softmax(-1).squeeze()
        answer_end_probs = answer_end_scores.softmax(-1).squeeze()

        # get the most likely beginning/end of answer with the argmax of the score
        answer_start_ix = torch.argmax(answer_start_scores)
        answer_end_ix = torch.argmax(answer_end_scores)

        answer_start = answer_start_ix
        answer_end = answer_end_ix + 1

        # print("answer locs:", answer_start, answer_end)

        # get input ids as list
        input_id_list = input_ids.squeeze().tolist()

        # detokenize the answer
        answer = tokenizer.decode(input_id_list[answer_start:answer_end])

        answer_start_prob_max = answer_start_probs[answer_start_ix].numpy()
        answer_end_prob_max = answer_end_probs[answer_end_ix].numpy()

        # answer_probs = [answer_start_prob_max, answer_end_prob_max]
        answer_probs = answer_start_prob_max * answer_end_prob_max

        # print(f"answer: {answer} ({answer_start}, {answer_end})")
        # print(
        #     f"answer probs: total {answer_probs}, start {answer_start_prob_max}, end {answer_end_prob_max}"
        # )

        expected_start_pos = labels["start_positions"][0]
        expected_end_pos = labels["end_positions"][0] + 1
        expected_answer = tokenizer.decode(
            input_id_list[expected_start_pos:expected_end_pos]
        )
        # print(
        #     f"expected answer: {expected_answer} ({expected_start_pos}, {expected_end_pos})"
        # )

        return SimpleNamespace(
            answer=answer,
            answer_locs=(answer_start, answer_end),
            answer_probs = (answer_start_prob_max, answer_end_prob_max),
            expected_answer=expected_answer,
            expected_answer_locs=(expected_start_pos, expected_end_pos),
        )

    # for each batch in val data, run predictions and compute metrics
    num_items = 0
    num_correct = 0
    num_incorrect = 0

    for idx, batch in enumerate(dataloader):
        # print(f"processing batch: {batch}")
        # run predict
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        bbox = batch["bbox"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        print('n start:', start_positions.shape)

        continue

        # with torch.no_grad():
        # outputs = model(
        #     input_ids=batch["input_ids"],
        #     attention_mask=batch["attention_mask"],
        #     bbox=batch["bbox"],
        #     pixel_values=batch["pixel_values"],
        # )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        # # log output
        # print(f"outputs: {outputs}")

        labels = {
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

        pred_results = check_preds(outputs, labels)
        # print(f"pred results: {pred_results}")

        # pretty print prediction info
        print(f'expected: {pred_results.expected_answer}\t\t\t({pred_results.expected_answer_locs})')
        print(f' predicted: {pred_results.answer}\t\t\t({pred_results.answer_locs})')
        print(f'  answer probs: start {pred_results.answer_probs[0]}, end {pred_results.answer_probs[1]}')

        is_correct = fuzzy(pred_results.answer, pred_results.expected_answer, 0.3)
        print('  correct? ', is_correct)

        print()

        num_items += 1
        if is_correct:
            num_correct += 1
        else:
            num_incorrect += 1

    # summarize results
    print(f"num items: {num_items}")
    print(f"num correct: {num_correct}")
    print(f"num incorrect: {num_incorrect}")
    print(f"accuracy: {num_correct / num_items:.2%}")


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
