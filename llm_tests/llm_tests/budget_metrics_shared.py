import typer
import os
import sys
import tempfile
import json
from typing import Optional
from types import SimpleNamespace
import importlib.util

import torch
import pandas as pd
from PIL import Image
import editdistance

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LayoutLMv3FeatureExtractor,
)
from llm_tests.modeling_llm3dec import LayoutLMv3Seq2SeqModel
from datasets import load_dataset, load_from_disk
from types import SimpleNamespace

def normalized_levenshtein(s1, s2):
    L = editdistance.eval(s1, s2)
    norm_div = max(len(s1), len(s2))
    return L / norm_div

