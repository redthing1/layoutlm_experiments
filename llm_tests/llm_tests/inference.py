import typer
import os
import sys
import tempfile
from typing import Optional
import importlib.util

import torch
import json
import pandas as pd
import pdfkit
import pdf2image

from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer, LayoutLMv3FeatureExtractor
from PIL import Image

def cli(
    model_path: str,
    feature_extractor: str,
    in_file: str,
    processor_id: str = "microsoft/layoutlmv3-base",
):
    _transformers_mod = __import__("transformers")

    # load the model
    print(f"loading model: {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(processor_id, apply_ocr=True)
    print(f"model loaded: {model_path}")

    # check if in file is url
    if in_file.startswith("http"):
        # open and print webpage
        print(f"opening webpage: {in_file}")
        tmp_pdf_out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        printed_webpage = pdfkit.from_in_file(in_file, tmp_pdf_out.name, options={ 'zoom': 0.5 })
        print(f"webpage printed: {in_file}")

        # process webpage
        print(f"processing document: {tmp_pdf_out.name}")
        # doc_img = Image.open(tmp_pdf_out.name).convert("RGB")
        doc_images = pdf2image.convert_from_path(tmp_pdf_out.name, dpi=200)
        doc_img = doc_images[0].convert("RGB")
    elif in_file.endswith(".pdf"):
        # open pdf as image
        print(f"opening pdf: {in_file}")
        doc_images = pdf2image.convert_from_path(in_file, dpi=200)
        doc_img = doc_images[0].convert("RGB")
    else:
        # try loading image directly
        print(f"opening image: {in_file}")
        doc_img = Image.open(in_file).convert("RGB")
    
    _feature_extractor_cls = getattr(_transformers_mod, feature_extractor)
    feature_extractor = _feature_extractor_cls(apply_ocr=True)
    doc_encoding = feature_extractor(doc_img, return_tensors="pt")
    # pixel_values = doc_encoding["pixel_values"]
    words = doc_encoding["words"]
    # boxes = doc_encoding["boxes"]

    print('detected words:', words)

    # ask user for questions
    user_question = typer.prompt("question:").strip()
    while user_question != "":
        # create model inputs
        encoding = processor(doc_img, user_question, truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        # print('detokenized input sequence:', tokenizer.decode(input_ids[0]))

        with torch.no_grad():
            outputs = model(**encoding)
        # print('model outputs:', outputs.keys())

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start_probs = answer_start_scores.softmax(-1).squeeze()
        answer_end_probs = answer_end_scores.softmax(-1).squeeze()

        # get the most likely beginning/end of answer with the argmax of the score
        answer_start_ix = torch.argmax(answer_start_scores)
        answer_end_ix = torch.argmax(answer_end_scores)

        answer_start = answer_start_ix
        answer_end = answer_end_ix + 1

        print('answer locs:', answer_start, answer_end)

        # get input ids as list
        input_id_list = input_ids.squeeze().tolist()

        # detokenize the answer
        answer = tokenizer.decode(input_id_list[answer_start:answer_end])

        print(f"answer: {answer}")

        answer_start_prob_max = answer_start_probs[answer_start_ix].numpy().item()
        answer_end_prob_max = answer_end_probs[answer_end_ix].numpy().item()

        # answer_probs = [answer_start_prob_max, answer_end_prob_max]
        answer_probs = answer_start_prob_max * answer_end_prob_max

        print(f'answer probs: total {answer_probs}, start {answer_start_prob_max}, end {answer_end_prob_max}')

        # ask user for another question
        user_question = typer.prompt("question:").strip()

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
