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

def normalized_levenshtein(s1, s2):
    L = editdistance.eval(s1, s2)
    norm_div = max(len(s1), len(s2))
    return L / norm_div

def cli(
    model_path: str,
    val_data_path: str,
    tiny_subset: bool = False,
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

    if tiny_subset:
        val_data = val_data.select(range(20))

    val_data_encoded = val_data.map(lambda x: x)
    val_data_encoded = val_data_encoded.remove_columns(["acceptable_answers", "acceptable_answers_starts", "acceptable_answers_ends"])
    val_data_encoded.set_format("torch")

    print('encoded val data:', val_data_encoded)
    print('encoded val data:', val_data_encoded.features)

    dataloader = torch.utils.data.DataLoader(val_data_encoded, batch_size=1)

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

        # expected_start_pos = labels["start_positions"][0]
        # expected_end_pos = labels["end_positions"][0] + 1
        # expected_answer = tokenizer.decode(
        #     input_id_list[expected_start_pos:expected_end_pos]
        # )
        # print(
        #     f"expected answer: {expected_answer} ({expected_start_pos}, {expected_end_pos})"
        # )

        acceptable_answers = []
        acceptable_answer_locs = []
        for item in zip(labels.acceptable_answers, labels.acceptable_answers_starts, labels.acceptable_answers_ends):
            # print('acceptable answer:', item)

            acc_answer_start = item[1]
            acc_answer_end = item[2] + 1

            acc_answer_text = tokenizer.decode(input_id_list[acc_answer_start:acc_answer_end])
            
            acceptable_answers.append(acc_answer_text)
            acceptable_answer_locs.append((acc_answer_start, acc_answer_end))

        return SimpleNamespace(
            answer=answer,
            answer_locs=(answer_start, answer_end),
            answer_probs = (answer_start_prob_max, answer_end_prob_max),
            acceptable_answers=acceptable_answers,
            acceptable_answer_locs=acceptable_answer_locs,
        )

    # for each batch in val data, run predictions and compute metrics
    num_items = 0
    num_exact_correct = 0

    total_normalized_levenshtein = 0

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

        acceptable_answers = val_data[idx]["acceptable_answers"]
        acceptable_answers_starts = val_data[idx]["acceptable_answers_starts"]
        acceptable_answers_ends = val_data[idx]["acceptable_answers_ends"]

        labels = SimpleNamespace(
            acceptable_answers=acceptable_answers,
            acceptable_answers_starts=acceptable_answers_starts,
            acceptable_answers_ends=acceptable_answers_ends,
        )

        pred_results = check_preds(outputs, labels)

        # pretty print prediction info
        print(f'expected: {pred_results.acceptable_answers}')
        print(f' predicted: {pred_results.answer}')
        print(f'  locs: {pred_results.answer_locs}')
        print(f'  probs: {pred_results.answer_probs}')

        if len(pred_results.acceptable_answers) == 0:
            # no answer
            continue

        # compute match validity
        best_similarity = 0
        best_answer_match = None

        pred_answer = pred_results.answer.strip().lower()
        
        for acceptable_answer in pred_results.acceptable_answers:
            acceptable_answer = acceptable_answer.strip().lower()
            
            # check fuzzy matching
            nl = normalized_levenshtein(acceptable_answer, pred_answer)
            
            # docvqa ANLS similarity metric
            tau = 0.5
            if nl <= tau:
                similarity = 1 - nl
            else:
                # nl is too big
                similarity = 0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer_match = acceptable_answer
        
        if best_answer_match:
            print(' matched answer:', best_answer_match)
        else:
            print(' no match')

        if abs(1 - best_similarity) < 0.001:
            num_exact_correct += 1
        
        total_normalized_levenshtein += best_similarity
        
        print(' best similarity:', best_similarity)

        num_items += 1

    # summarize results
    print(f"num items: {num_items}")
    print(f"num exact match: {num_exact_correct}")
    print(f"accuracy: {num_exact_correct / num_items:.2%}")
    anls = total_normalized_levenshtein / num_items
    print(f"anls: {anls:.3f}")


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
