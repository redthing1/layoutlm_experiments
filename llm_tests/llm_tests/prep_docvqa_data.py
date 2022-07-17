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
from datasets import load_dataset, load_from_disk
from PIL import Image
from sacremoses import MosesDetokenizer
import editdistance

from llm_tests.ocr import MicrosoftReadOCR

Models = namedtuple("Models", "processor model")

MODEL_ID = "microsoft/layoutlmv3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ROOT_DIR = None


def load_models_from_hf():
    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)

    return Models(processor, model)


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def fuzzy(s1, s2, threshold=0.2):
    return (editdistance.eval(s1, s2) / ((len(s1) + len(s2)) / 2)) < threshold

def fuzzy_diff(s1, s2):
    return (editdistance.eval(s1, s2) / ((len(s1) + len(s2)) / 2))

def extract_ocr_words_boxes(root_dir, feature_extractor, ocr_engine, examples):
    if ocr_engine == 'dataset':
        return load_dataset_ocr_words_and_boxes(root_dir, feature_extractor, examples)
    elif ocr_engine == 'microsoft':
        return get_msread_ocr_words_and_boxes(root_dir, feature_extractor, examples)
    elif ocr_engine == 'tesseract':
        return get_ocr_words_and_boxes(root_dir, feature_extractor, examples)
    else:
        assert False, "Unknown ocr engine"


def get_ocr_words_and_boxes(root_dir, feature_extractor, examples):
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


def load_dataset_ocr_words_and_boxes(root_dir, feature_extractor, examples):
    # load imgaes
    images = [
        Image.open(f"{root_dir}/{image_file}").convert("RGB")
        for image_file in examples["image"]
    ]

    # load ocrs
    ocrs = []
    for image_file in examples["image"]:
        ocr_file = image_file.replace("documents/", "ocr_results/").replace(
            ".png", ".json"
        )
        with open(f"{root_dir}/{ocr_file}") as f:
            ocr = json.load(f)
        ocrs.append(ocr)

    encoded_inputs = feature_extractor(images)
    examples["pixel_values"] = encoded_inputs.pixel_values

    # save ocr words and boxes
    batch_words = []
    batch_boxes = []

    for i, ocr in enumerate(ocrs):
        img = images[i]
        words = []
        boxes = []
        for doc in ocr["recognitionResults"]:
            for line in doc["lines"]:
                for w in line["words"]:
                    points = w["boundingBox"]
                    words.append(w["text"])
                    x1 = min(points[0::2])
                    x2 = max(points[0::2])
                    y1 = min(points[1::2])
                    y2 = max(points[1::2])
                    boxes.append(
                        normalize_bbox((x1, y1, x2, y2), img.width, img.height)
                    )
        batch_words.append(words)
        batch_boxes.append(boxes)

    examples["words"] = batch_words
    examples["boxes"] = batch_boxes

    return examples

MS_OCR_INSTANCE = None

def get_msread_ocr_words_and_boxes(root_dir, feature_extractor, examples):
    global MS_OCR_INSTANCE

    # load imgaes
    images = [
        Image.open(f"{root_dir}/{image_file}").convert("RGB")
        for image_file in examples["image"]
    ]

    # load ocrs
    if MS_OCR_INSTANCE is None:
        MS_OCR_INSTANCE = MicrosoftReadOCR()
    ms_ocr = MS_OCR_INSTANCE
    ocrs = []
    for image_file in examples["image"]:
        # retry 3 times on exception
        attempt = 0
        while attempt < 3:
            try:
                print('requesting ms read ocr for', image_file)
                ocr_results = ms_ocr.analyze_file(f"{root_dir}/{image_file}")
                ocrs.append(ocr_results)
                break
            except Exception as e:
                print(f'Error requesting ms read ocr for {image_file}: {e}')
                attempt += 1
                if attempt >= 3:
                    raise e

    encoded_inputs = feature_extractor(images)
    examples["pixel_values"] = encoded_inputs.pixel_values

    # save ocr words and boxes
    batch_words = []
    batch_boxes = []

    for i, ocr in enumerate(ocrs):
        img = images[i]
        words = []
        boxes = []
        for item in ocr:
            word = item[0]
            box = item[1]
            
            # print('processing word: ', word, box)
            boxes.append(normalize_bbox(box, img.width, img.height))
            words.append(word)

        batch_words.append(words)
        batch_boxes.append(boxes)

    examples["words"] = batch_words
    examples["boxes"] = batch_boxes

    return examples


def better_subfinder(words_list, answer_query, try_hard=True):
    matches = []
    start_indices = []
    end_indices = []
    
    detokenizer = MosesDetokenizer(lang="en")
    
    # first try dumber, faster method, but this has false negatives
    answer_list = answer_query.split()
    for idx, i in enumerate(range(len(words_list))):
        # if (
        #     words_list[i] == answer_list[0]
        #     and words_list[i : i + len(answer_list)] == answer_list
        # ):
        if len(words_list[i : i + len(answer_list)]) == len(answer_list) and all(
            fuzzy(words_list[i + j], answer_list[j]) for j in range(len(answer_list))
        ):
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)

    if matches:
        return matches[0], start_indices[0], end_indices[0]

    if not try_hard:
        # fail
        return None, 0, 0
    
    # if that failed, use our stronger method to find missed matches
    smart_matches = []
    for start_pos in range(len(words_list)):
        for end_pos in range(start_pos, len(words_list)):
            # use a length heuristic
            n_pieces = end_pos - start_pos + 1

            # check that the n pieces is close to the length of the answer list
            if (
                abs(n_pieces - len(answer_list)) > 5
                or n_pieces < len(answer_list) / 2
                or n_pieces > len(answer_list) * 2
            ):
                continue

            piece = words_list[start_pos:end_pos+1]
            # print('checking piece:', piece)

            # try to detokenize
            detok_piece = detokenizer.detokenize(piece)
            # print(' detok piece:', detok_piece)

            # check if this piece is close to the answer
            diff = fuzzy_diff(detok_piece, answer_query)
            if diff < 0.2:
                smart_matches.append((piece, diff, start_pos, end_pos))

            if diff == 0:
                break # perfect match, no need to continue
    
    if smart_matches:
        # sort smart matches by diff
        best_match = sorted(smart_matches, key=lambda x: x[1])[0]

        return best_match[0], best_match[2], best_match[3]
    
    # fail
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
        print(" Expected answer:", answers[batch_index])
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            match, word_idx_start, word_idx_end = better_subfinder(
                words_example, answer.lower()
            )
            print('searching for', answer, 'in', words_example)
            if match:
                print(
                    " Found match (standard):",
                    match,
                    "from",
                    word_idx_start,
                    "to",
                    word_idx_end,
                )
                # print(" Verbose: words_example:", words_example, 'answer:', answer.lower().split())
                # print(" Match is from ", words_example[word_idx_start], "to", words_example[word_idx_end])
                break
        # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
        if not match:
            print("Trying to recover from mismatch")
            for answer in answers[batch_index]:
                for i in range(len(answer)):
                    if len(answer) == 1:
                        # this method won't work for single-character answers
                        print(" Skipping single-character answer")
                        break
                    # drop the ith character from the answer
                    answer_i = answer[:i] + answer[i + 1 :]
                    # print('Trying: ', i, answer, answer_i, answer_i.lower().split())
                    # check if we can find this one in the context
                    match, word_idx_start, word_idx_end = better_subfinder(
                        words_example, answer_i.lower(), try_hard=False
                    )
                    if match:
                        print(
                            " Found match (truncated):",
                            match,
                            "from",
                            word_idx_start,
                            "to",
                            word_idx_end,
                        )
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
            print(
                "sliced word ids from",
                token_start_index,
                "to",
                token_end_index + 1,
                "out of",
                0,
                len(encoding.word_ids(batch_index)),
            )
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
                    print(" start:", token_start_index)
                    found_start = True
                    break
                else:
                    token_start_index += 1
                    # print(' start id did not match:', id, word_idx_start)

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    print(" end:", token_end_index)
                    found_end = True
                    break
                else:
                    token_end_index -= 1
                    # print(' end id did not match:', id, word_idx_end)

            if found_end and found_start:
                start_positions.append(token_start_index)
                end_positions.append(token_end_index)
                print(
                    "Verifying start position and end position:",
                    batch_index,
                    start_positions,
                    end_positions,
                )
                print("True answer:", answer)
                start_position = start_positions[batch_index]
                end_position = end_positions[batch_index]
                reconstructed_answer = tokenizer.decode(
                    encoding.input_ids[batch_index][start_position : end_position + 1]
                )
                print("Reconstructed answer:", reconstructed_answer)
                print("-----------")
            else:
                print(
                    "could not find start or end positions, probably it was truncated out"
                )
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
    save_ocr: str = None,
    ocr_engine: str = "dataset",
    tiny_subset: bool = False,
    resume_from_ocr: str = None,
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

    dataset = None
    if not tiny_subset:
        dataset = Dataset.from_pandas(df)
    else:
        # dataset = Dataset.from_pandas(df.sample(n=8))
        dataset = Dataset.from_pandas(df.iloc[:8])
    print(f"dataset size: {len(dataset)}")

    if resume_from_ocr:
        # load ocr data
        print("loading ocr data from", resume_from_ocr)
        dataset_with_ocr = load_from_disk(resume_from_ocr)
    else:
        print("extracting ocr words and boxes")
        feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=(ocr_engine == "tesseract"))
        run_ocr_map_func = lambda x: extract_ocr_words_boxes(
            root_dir=ROOT_DIR,
            feature_extractor=feature_extractor,
            ocr_engine=ocr_engine,
            examples=x,
        )
        dataset_with_ocr = dataset.map(run_ocr_map_func, batched=True, batch_size=1)
        print(dataset_with_ocr)
        print(f"dataset with ocr keys: {dataset_with_ocr.features}")

        if save_ocr:
            print("saving ocr data to", save_ocr)
            dataset_with_ocr.save_to_disk(save_ocr)

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
        batch_size=1,
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
