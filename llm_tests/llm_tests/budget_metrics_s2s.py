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
    AutoTokenizer,
    LayoutLMv3FeatureExtractor,
)
from llm_tests.modeling_llm3dec import LayoutLMv3Seq2SeqModel
from datasets import load_dataset, load_from_disk

def normalized_levenshtein(s1, s2):
    L = editdistance.eval(s1, s2)
    norm_div = max(len(s1), len(s2))
    return L / norm_div

def cli(
    model_path: str,
    val_data_path: str,
    decoder_tokenizer_id: str,
    tiny_subset: bool = False,
    device: Optional[str] = None,
):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    print(f"loading model: {model_path}")
    model = LayoutLMv3Seq2SeqModel.from_pretrained(model_path, ignore_mismatched_sizes=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)
    print(f"model loaded: {model_path}")

    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer_id)

    print(f"loading val data: {val_data_path}")
    val_data = load_from_disk(val_data_path)
    print(f"val data loaded: {val_data_path}")

    if tiny_subset:
        val_data = val_data.select(range(min(20, len(val_data))))

    val_data_encoded = val_data.map(lambda x: x)
    val_data_encoded = val_data_encoded.remove_columns(["acceptable_answers"])
    val_data_encoded.set_format("torch")

    print('encoded val data:', val_data_encoded)
    print('encoded val data:', val_data_encoded.features)

    dataloader = torch.utils.data.DataLoader(val_data_encoded, batch_size=1)

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

        labels = batch["labels"].to(device)
        acceptable_answers = val_data[idx]["acceptable_answers"]

        # outputs = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     bbox=bbox,
        #     pixel_values=pixel_values,
        #     labels=labels,
        # )

        # # # log output
        # # print(f"outputs: {outputs}")

        # print(f"model outputs: {outputs.keys()}")
        # # predicted_answer = decoder_tokenizer.decode(outputs, skip_special_tokens=True)
        # # print(f"predicted answer: {predicted_answer}")
        # assert False, "bean and cheese"

        # print('decoded input_ids:', tokenizer.decode(input_ids[0], skip_special_tokens=False))
        # print('decoded labels:', decoder_tokenizer.decode(labels[0], skip_special_tokens=False))

        gen_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            do_sample=True,
            top_p=0.9,
            temperature=0.2,            
        )
        print(f"gen_output: {gen_output}")

        predicted_answer = decoder_tokenizer.decode(gen_output[0], skip_special_tokens=True)
        print(f"predicted answer: {predicted_answer}")

        # pretty print prediction info
        print(f'expected: {acceptable_answers}')
        print(f' predicted: {predicted_answer}')

        if len(acceptable_answers) == 0:
            # no answer
            continue

        # compute match validity
        best_similarity = 0
        best_answer_match = None

        pred_check_answer = predicted_answer.strip().lower()
        
        for acceptable_answer in acceptable_answers:
            acceptable_answer = acceptable_answer.strip().lower()
            
            # check fuzzy matching
            nl = normalized_levenshtein(acceptable_answer, pred_check_answer)
            
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
    print(f"exact acc: {num_exact_correct / num_items:.2%}")
    anls = total_normalized_levenshtein / num_items
    print(f"anls: {anls:.3f}")


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
