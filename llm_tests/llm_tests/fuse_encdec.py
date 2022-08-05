from distutils.command.config import config
import typer
import os
from typing import Optional, Dict, Tuple
import collections
import json
import logging

import torch
import pandas as pd
import numpy as np

from transformers import LayoutLMv3Model, AutoModelForCausalLM, EncoderDecoderModel
from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer

from llm_tests.modeling_llm3dec import LayoutLMv3Seq2SeqModel

from tqdm.auto import tqdm

# from llm_tests.qa_trainer import QuestionAnsweringTrainer

logger = logging.getLogger(__name__)


def cli(
    encoder_model_path: str,
    decoder_model_path: str,
    save_to: str = None,
    test_input: str = None,
):
    # load the encoder model
    print(f"loading encoder model: {encoder_model_path}")
    model_encoder = LayoutLMv3Model.from_pretrained(encoder_model_path)
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=True
    )
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_path)
    print(f"encoder model loaded: {encoder_model_path}")

    # load the decoder model
    print(f"loading decoder model: {decoder_model_path}")
    model_decoder = AutoModelForCausalLM.from_pretrained(decoder_model_path)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_path)
    print(f"decoder model loaded: {decoder_model_path}")

    # # fuse the models
    # print("fusing models")
    # seq2seq_model = EncoderDecoderModel(encoder=model_encoder, decoder=model_decoder)
    # print("models fused")
    # print(seq2seq_model)

    # fuse the models
    print(f"fusing models {encoder_model_path} and {decoder_model_path}")
    # seq2seq_model = LayoutLMv3Seq2SeqModel.from_encoder_decoder_pretrained(
    #     encoder_model_path, decoder_model_path
    # )
    # model_decoder.config.max_position_embeddings = 1024
    # model_encoder.config.max_position_embeddings = 1024
    # print(f'position embedding size: encoder: {model_encoder.config.max_position_embeddings}, decoder: {model_decoder.config.max_position_embeddings}')
    seq2seq_model = LayoutLMv3Seq2SeqModel(encoder=model_encoder, decoder=model_decoder)

    # seq2seq_model.config.decoder.max_position_embeddings = 512
    # seq2seq_model.config.encoder.max_position_embeddings = 512

    # decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    # seq2seq_model.config.pad_token_id = decoder_tokenizer.pad_token_id
    # seq2seq_model.config.pad_token_id = decoder_tokenizer.cls_token_id

    seq2seq_model.config.decoder_start_token_id = decoder_tokenizer.cls_token_id
    seq2seq_model.config.pad_token_id = decoder_tokenizer.pad_token_id

    print("models fused")
    print(seq2seq_model)
    print(seq2seq_model.config)

    if save_to is not None:
        print(f"saving model to {save_to}")
        seq2seq_model.save_pretrained(save_to)
        print(f"model saved to {save_to}")

    if test_input is not None:
        # test fused model
        print("testing model")

        from PIL import Image

        doc_img = Image.open(test_input).convert("RGB")

        encoder_encoding = processor(
            doc_img,
            "What is the document about?",
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        decoder_labels = decoder_tokenizer(
            "Document",
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        print("encoder_encoding:", encoder_encoding)
        print("decoder_labels:", decoder_labels)

        print(f'attention mask shapes: encoder: {encoder_encoding.attention_mask.shape}, decoder: {decoder_labels.attention_mask.shape}')

        # outputs = model(
        #     input_ids=encoding.input_ids,
        #     attention_mask=encoding.attention_mask,
        #     bbox=encoding.bbox,
        #     pixel_values=encoding.pixel_values,
        # )
        outputs = seq2seq_model(
            input_ids=encoder_encoding.input_ids,
            labels=decoder_labels.input_ids,
            attention_mask=encoder_encoding.attention_mask,
            bbox=encoder_encoding.bbox,
            pixel_values=encoder_encoding.pixel_values,
        )
        print("model output:", outputs)


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
