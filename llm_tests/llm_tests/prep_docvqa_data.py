import typer
import os
from typing import Optional
from collections import namedtuple

import torch
from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import LayoutLMv3FeatureExtractor
import json
import pandas as pd
from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from datasets import load_dataset
from PIL import Image

Models = namedtuple("Models", "processor model")

MODEL_ID = "microsoft/layoutlmv3-base"
feature_extractor = LayoutLMv3FeatureExtractor()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ROOT_DIR = None


def load_models_from_hf():
    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)

    return Models(processor, model)


def get_ocr_words_and_boxes(root_dir, examples):
    # get a batch of document images
    images = [
        Image.open(f"{root_dir}/{image_file}").convert("RGB")
        for image_file in examples["image"]
    ]

    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    encoded_inputs = feature_extractor(images)

    examples["pixel_values"] = encoded_inputs.pixel_values
    examples["words"] = encoded_inputs.words
    examples["boxes"] = encoded_inputs.boxes

    return examples


# source: https://stackoverflow.com/a/12576755
def subfinder(words_list, answer_list):
    if len(answer_list) == 0:
        assert False, "Answer list is empty"
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if (
            words_list[i] == answer_list[0]
            and words_list[i : i + len(answer_list)] == answer_list
        ):
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def encode_dataset(examples, max_length=512):
    # take a batch
    questions = examples["question"]
    words = examples["words"]
    boxes = examples["boxes"]

    # encode it
    encoding = tokenizer(
        questions,
        words,
        boxes,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # next, add start_positions and end_positions
    start_positions = []
    end_positions = []
    answers = examples["answers"]
    # for every example in the batch:
    for batch_index in range(len(answers)):
        print("Batch index:", batch_index)
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            match, word_idx_start, word_idx_end = subfinder(
                words_example, answer.lower().split()
            )
            if match:
                print(" Found match (standard):", match, 'from', word_idx_start, 'to', word_idx_end)
                # print(" Verbose: words_example:", words_example, 'answer:', answer.lower().split())
                # print(" Match is from ", words_example[word_idx_start], "to", words_example[word_idx_end])
                break
        # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
        if not match:
            print('Trying to recover from mismatch')
            for answer in answers[batch_index]:
                for i in range(len(answer)):
                    if len(answer) == 1:
                        # this method won't work for single-character answers
                        print(" Skipping single-character answer")
                        break
                    # drop the ith character from the answer
                    answer_i = answer[:i] + answer[i+1:]
                    # print('Trying: ', i, answer, answer_i, answer_i.lower().split())
                    # check if we can find this one in the context
                    match, word_idx_start, word_idx_end = subfinder(
                        words_example, answer_i.lower().split()
                    )
                    if match:
                        print(' Found match (truncated):', match, 'from', word_idx_start, 'to', word_idx_end)
                        break
        # END OF EXPERIMENT

        if match:
            sequence_ids = encoding.sequence_ids(batch_index)
            # Start token index of the current span in the text.
            token_start_index = 0
            # skip <pad> tokens
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(encoding.input_ids[batch_index]) - 1
            # skip <pad> tokens
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            word_ids = encoding.word_ids(batch_index)[
                token_start_index : token_end_index + 1
            ]
            print('sliced word ids from', token_start_index, 'to', token_end_index + 1,
                'out of', 0, len(encoding.word_ids(batch_index)))
            # print('trying to match start and end tokens:', word_ids, word_idx_start, word_idx_end)
            # decoded_words = tokenizer.decode(
            #     encoding.input_ids[batch_index][token_start_index : token_end_index + 1]
            # )
            # print('decoded_words:', decoded_words)
            # all_words = tokenizer.decode(encoding.input_ids[batch_index])
            # print('all_words:', all_words)
            found_start = False
            found_end = False
            for id in word_ids:
                if id == word_idx_start:
                    start_positions.append(token_start_index)
                    print(' start:', token_start_index)
                    found_start = True
                    break
                else:
                    token_start_index += 1
                    # print(' start id did not match:', id, word_idx_start)

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    end_positions.append(token_end_index)
                    print(' end:', token_end_index)
                    found_end = True
                    break
                else:
                    token_end_index -= 1
                    # print(' end id did not match:', id, word_idx_end)
            
            if found_end and found_start:
                print("Verifying start position and end position:", batch_index, start_positions, end_positions)
                print("True answer:", answer)
                start_position = start_positions[batch_index]
                end_position = end_positions[batch_index]
                reconstructed_answer = tokenizer.decode(
                    encoding.input_ids[batch_index][start_position : end_position + 1]
                )
                print("Reconstructed answer:", reconstructed_answer)
                print("-----------")
            else:
                print('could not find start or end positions, probably it was truncated out')
                start_positions.append(cls_index)
                end_positions.append(cls_index)
        else:
            print("Answer not found in context")
            print("-----------")
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    encoding["pixel_values"] = examples["pixel_values"]
    encoding["start_positions"] = start_positions
    encoding["end_positions"] = end_positions

    return encoding


def cli(
    dataset_dir: str,
    dataset_split: str,
    out_dir: str,
):
    global ROOT_DIR
    ROOT_DIR = dataset_dir

    # models = load_models_from_hf()
    # print('loaded models')

    with open(f"{dataset_dir}/{dataset_split}_v1.0.json") as f:
        data = json.load(f)

    print(f"loaded data info: {data.keys()}")

    print("dataset:", data["dataset_name"])
    print("dataset split:", data["dataset_split"])

    df = pd.DataFrame(data["data"])
    print("data preview:", df.head())

    # dataset = Dataset.from_pandas(df)
    dataset = Dataset.from_pandas(df)
    print(f"dataset size: {len(dataset)}")

    print("extracting ocr words and boxes")
    run_ocr_map_func = lambda x: get_ocr_words_and_boxes(ROOT_DIR, x)
    dataset_with_ocr = dataset.map(
        run_ocr_map_func,
        batched=True,
        batch_size=2)
    print(dataset_with_ocr)
    print(f"dataset with ocr keys: {dataset_with_ocr.features}")

    # encode the dataset
    print("encoding dataset")
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "attention_mask": Sequence(Value(dtype="int64")),
            # 'token_type_ids': Sequence(Value(dtype='int64')),
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
            "start_positions": Value(dtype="int64"),
            "end_positions": Value(dtype="int64"),
        }
    )

    encoded_dataset = dataset_with_ocr.map(
        encode_dataset,
        batched=True,
        batch_size=2,
        remove_columns=dataset_with_ocr.column_names,
        features=features,
    )

    print(f"encoded dataset: {encoded_dataset}")

    # now save out the dataset
    print(f"saving dataset to {out_dir}")
    encoded_dataset.save_to_disk(out_dir)

    print("done")


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
