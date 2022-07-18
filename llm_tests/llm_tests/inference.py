import typer
import os
from typing import Optional

import torch
import json
import pandas as pd

from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer

def cli(
    model_path: str,
    processor_id: str = "microsoft/layoutlmv3-base",
):
    # load the model
    print(f"loading model: {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(processor_id, apply_ocr=True)
    print(f"model loaded: {model_path}")

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
