import typer
import os
from typing import Optional, Dict, Tuple
import collections
import json
import logging

import torch
import pandas as pd
import numpy as np

from transformers import  LayoutLMv3Model, AutoModelForCausalLM, EncoderDecoderModel
from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer

from tqdm.auto import tqdm
# from llm_tests.qa_trainer import QuestionAnsweringTrainer

logger = logging.getLogger(__name__)

def cli(
    encoder_model_path: str,
    decoder_model_path: str,
    save_to: str = None,
):
    # load the encoder model
    print(f"loading encoder model: {encoder_model_path}")
    # model_encoder = LayoutLMv3Model.from_pretrained(encoder_model_path)
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_path)
    print(f"encoder model loaded: {encoder_model_path}")

    # load the decoder model
    print(f"loading decoder model: {decoder_model_path}")
    # model_decoder = AutoModelForCausalLM.from_pretrained(decoder_model_path)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_path)
    print(f"decoder model loaded: {decoder_model_path}")

    # # fuse the models
    # print("fusing models")
    # seq2seq_model = EncoderDecoderModel(encoder=model_encoder, decoder=model_decoder)
    # print("models fused")
    # print(seq2seq_model)

    # fuse the models
    print(f"fusing models {encoder_model_path} and {decoder_model_path}")
    seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_path, decoder_model_path)
    
    # decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    # seq2seq_model.config.pad_token_id = decoder_tokenizer.pad_token_id
    # seq2seq_model.config.pad_token_id = decoder_tokenizer.cls_token_id

    seq2seq_model.config.decoder_start_token_id = decoder_tokenizer.cls_token_id
    seq2seq_model.config.pad_token_id = decoder_tokenizer.pad_token_id

    print("models fused")
    print(seq2seq_model)

    if save_to is not None:
        print(f"saving model to {save_to}")
        seq2seq_model.save_pretrained(save_to)
        print(f"model saved to {save_to}")

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
