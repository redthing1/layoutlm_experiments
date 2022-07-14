import typer
import os
from typing import Optional
from collections import namedtuple
# from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer, AutoTokenizer, T5ForConditionalGeneration

Models = namedtuple('Models', 'tokenizer model')

def load_models_from_hf():
    # model_name = "SEBIS/code_trans_t5_base_source_code_summarization_python_multitask_finetune"
    model_name = "Salesforce/codet5-small"

    # load tokenizer and model
    # tokenizer_class = T5TokenizerFast
    tokenizer_class = RobertaTokenizer
    model_class = T5ForConditionalGeneration

    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)

    return Models(tokenizer, model)


def load_models_from_disk(model_path):
    print(f"Loading model from {model_path}")
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    return Models(tokenizer, model)


def test_inference(models, text):
    input_ids = models.tokenizer(text, return_tensors="pt").input_ids

    # try decoding inputs
    in_decoded = [models.tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids[0]]
    print('tokenization: ', in_decoded)

    # simply generate a single sequence
    generated_ids = models.model.generate(input_ids, max_length=128)
    out_seqs = [models.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]

    return out_seqs[0]


def bean_cli():
    # models = load_models_from_hf()
    models = load_models_from_disk(os.environ.get("MODEL", 'model'))

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
