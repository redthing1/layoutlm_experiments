import typer
import os
from typing import Optional
from collections import namedtuple

from transformers import AutoProcessor, AutoModelForQuestionAnswering
from datasets import load_dataset
import torch
from transformers import LayoutLMv3FeatureExtractor
from PIL import Image

Models = namedtuple('Models', 'processor model')

MODEL_ID = "microsoft/layoutlmv3-base"

def load_models_from_hf():
    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)

    return Models(processor, model)


def load_models_from_disk(model_path):
    print(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    return Models(processor, model)


def test_inference(models, document, question):
    # open image
    image = Image.open(document).convert("RGB")

    # extract features
    feature_extractor = LayoutLMv3FeatureExtractor()
    doc_encoding = feature_extractor(image, return_tensors="pt")
    # print(encoding.keys())

    # encode with question
    question_encoding = models.processor(image, question, words=doc_encoding["tokens"], boxes=doc_encoding["bboxes"], return_tensors="pt")
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])

    # outputs = models.model(**encoding)
    outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    pass


def bean_cli():
    models = load_models_from_hf()
    # models = load_models_from_disk(os.environ.get("MODEL", 'model'))

    print('ready.\n')

    while text := multiline_in():
        print('\ngenerating\n')
        text = text.replace(r'\\n', r'\n')
        seq = test_inference(models, text)
        print(seq)


def multiline_in(prompt=''):
    print(prompt, end='')
    import sys
    return sys.stdin.read()

def main():
    typer.run(bean_cli)


if __name__ == "__main__":
    main()
