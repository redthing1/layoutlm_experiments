import typer
import os
import sys
import time
import tempfile
from typing import Optional
import importlib.util

import torch
import json
import pandas as pd
import pdfkit
import pdf2image

from transformers import AutoProcessor, AutoTokenizer, LayoutLMv3FeatureExtractor
from llm_tests.modeling_llm3dec import LayoutLMv3Seq2SeqModel
from PIL import Image

from llm_tests.segify_ocr import segify_boxes

def cli(
    model_path: str,
    feature_extractor: str,
    in_file: str,
    processor_id: str = "microsoft/layoutlmv3-base",
    segify: bool = True,
):
    _transformers_mod = __import__("transformers")

    # load the model
    start_time = time.time()
    print(f"loading model: {model_path}")
    model = LayoutLMv3Seq2SeqModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(processor_id, apply_ocr=True)
    elapsed_time = time.time() - start_time
    print(f"model loaded in {elapsed_time:.2f}s: {model_path}")

    # check if in file is url
    start_time = time.time()
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
    elapsed_time = time.time() - start_time
    print(f"document loaded in {elapsed_time:.2f}s: {in_file}")
    
    _feature_extractor_cls = getattr(_transformers_mod, feature_extractor)
    feature_extractor = _feature_extractor_cls(apply_ocr=True)
    
    print(f"extracting features...")
    start_time = time.time()
    doc_encoding = feature_extractor(doc_img, return_tensors="pt")
    # pixel_values = doc_encoding["pixel_values"]
    words = doc_encoding["words"]
    old_boxes = doc_encoding.boxes

    if not segify:
        # segify the boxes
        print("segifying boxes")
        doc_encoding.boxes = new_boxes = segify_boxes(old_boxes=old_boxes, words=words, row_diff=8, col_diff=40)
    
    elapsed_time = time.time() - start_time
    print(f"features extracted in {elapsed_time:.2f}s")

    # # dump boxes
    # print("old boxes:", old_boxes)
    # print("new boxes:", doc_encoding.boxes)

    print('detected words:', words)

    # ask user for questions
    user_question = typer.prompt("question:").strip()
    while user_question != "":
        # create model inputs
        encoding = processor(doc_img, user_question, truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        # print('detokenized input sequence:', tokenizer.decode(input_ids[0]))

        # # bbox vs box
        # print("bbox:", encoding.bbox)
        # print("box:", doc_encoding.boxes)

        # print('encoding:', encoding.keys())
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids,
                attention_mask=encoding.attention_mask,
                # bbox=doc_encoding.boxes,
                bbox=encoding.bbox,
                pixel_values=encoding.pixel_values,

                do_sample=True,
                # top_p=0.95,
                # temperature=0.1,
                top_p=0.9,
                temperature=0.1,
            )

        # detokenize the answer
        answer = tokenizer.decode(output_seq[0], skip_special_tokens=True)

        print(f"answer: {answer}")

        # ask user for another question
        user_question = typer.prompt("question:").strip()

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
