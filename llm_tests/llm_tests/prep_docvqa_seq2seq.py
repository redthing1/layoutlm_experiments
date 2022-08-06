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
# DECODER_MODEL_ID = "bert-base-cased"
decoder_tokenizer = None
# tokenizer.pad_token = tokenizer.eos_token
# decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
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
        # return_token_type_ids=True,
    )

    answers = examples["answers"]
    decoder_labels = []
    decoder_attention_masks = []

    global decoder_tokenizer
    def decoder_tokenize(data):
        return decoder_tokenizer(
            data,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    # print('examples:', answers)

    found_acceptable_answer = False

    batch_acceptable_answers = []

    # for every example in the batch:
    for batch_index in range(len(questions)):
        acceptable_answers = []

        print()
        print("-----------")
        print("Batch index:", batch_index)
        print(" Expected answers:", answers[batch_index])
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            print('searching for', answer, 'in', words_example)
            
            # check that the answer contains most of the words in the example
            answer_words = answer.split()
            answer_words = [word.lower() for word in answer_words]

            match = False

            # ensure that some of the answer words contain the example words
            for answer_word in answer_words:
                for word in words_example:
                    if word in answer_word:
                        match = True
                        break

            if match:
                acceptable_answers.append(answer)

                # check if alredy found an acceptable answer
                if found_acceptable_answer:
                    print("already found acceptable answer")
                    continue # go to next answer

                found_acceptable_answer = True
                print("true answer:", answer)

                # store decoder result
                decoder_encoding = decoder_tokenize(answer)
                decoder_labels.append(decoder_encoding.input_ids)
                decoder_attention_masks.append(decoder_encoding.attention_mask)

                # detokenize to make sure
                decoder_answer = decoder_tokenizer.decode(decoder_encoding.input_ids, skip_special_tokens=True)
                print("decoder answer:", decoder_answer)
            else:
                print(
                    "could not find expected answer, probably it was truncated out"
                )

    if not found_acceptable_answer:
        print("answer not found in context")

        decoder_encoding = decoder_tokenize('answer not found in context')
        decoder_labels.append(decoder_encoding.input_ids)
        decoder_attention_masks.append(decoder_encoding.attention_mask)

        # # detokenize to make sure
        # decoder_answer = decoder_tokenizer.decode(decoder_encoding.input_ids, skip_special_tokens=False)
        # print("decoder answer:", decoder_answer)
    
    # add acceptable answers to the batch lists
    batch_acceptable_answers.append(acceptable_answers)

    # # sanity checking
    # assert len(start_positions) == len(questions), f"start_positions and questions are different lengths: {len(start_positions)} vs {len(questions)}"
    # assert len(end_positions) == len(questions), f"end_positions and questions are different lengths: {len(end_positions)} vs {len(questions)}"

    encoding["pixel_values"] = examples["pixel_values"]
    # decoder_labels[decoder_labels[:, :] == decoder_tokenizer.pad_token_id] = -100
    
    # print('eos: ', tokenizer.eos_token_id, 'pad: ', tokenizer.pad_token_id)
    # print('decoder_labels:', decoder_labels)
    # print('attention_mask:', encoding.attention_mask)
    # print('decoder_attention_masks:', decoder_attention_masks)
    
    # replace decoder pad tokens with -100 to ignore them when calculating loss
    exc_labels = [
        [-100 if mask == 0 else token for mask, token in mask_and_tokens]
        for mask_and_tokens in [
            zip(masks, labels)
            for masks, labels in zip(decoder_attention_masks, decoder_labels)
        ]
    ]
    # print('exc_labels:', exc_labels)
    # encoding["labels"] = [decoder_ids]
    encoding["labels"] = exc_labels
    encoding["decoder_attention_mask"] = decoder_attention_masks
    

    # store acceptable answers
    encoding["acceptable_answers"] = batch_acceptable_answers

    return encoding


def cli(
    dataset_dir: str,
    dataset_split: str,
    out_dir: str,
    save_ocr: str = None,
    ocr_engine: str = "dataset",
    tiny_subset: bool = False,
    decoder_model: str = None,
    resume_from_ocr: str = None,
    procs: int = 1,
):
    global ROOT_DIR
    ROOT_DIR = dataset_dir

    if decoder_model is None:
        assert False, "decoder_model is required"
    
    global decoder_tokenizer
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)

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
        # dataset = Dataset.from_pandas(df.iloc[8:10])
        dataset = Dataset.from_pandas(df[:8])
    
    print(f"dataset size: {len(dataset)}")
    print(f"dataset features: {dataset.features}")

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
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
            "labels": Sequence(feature=Value(dtype="int64")),
            "decoder_attention_mask": Sequence(Value(dtype="int64")),

            "acceptable_answers": Sequence(feature=Value(dtype='string', id=None)),
        }
    )

    encoded_dataset = dataset_with_ocr.map(
        encode_dataset,
        batched=True,
        batch_size=1,
        remove_columns=dataset_with_ocr.column_names,
        features=features,
        num_proc=procs,
    )

    print(f"encoded dataset: {encoded_dataset}")

    # now save out the dataset
    print(f"saving dataset to {out_dir}")
    encoded_dataset.save_to_disk(out_dir)

    # try collecting some stats
    # count how many are missing (CLS/failure)

    eos = decoder_tokenizer.eos_token_id
    # print('enclab', encoded_dataset[4]["labels"], "eos", eos)
    failed_matches = encoded_dataset.filter(lambda x: x["labels"][1] == eos)
    print()
    print('failed matches:', len(failed_matches))
    reconst_ratio = (1 - len(failed_matches) / len(encoded_dataset))
    print(f'successfully reconstructed: {reconst_ratio:.2%}')

    print("done")


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
