# layoutlm_experiments

## finetune on docvqa

### 1. process docvqa data for LayoutLMv3

#### not recommended: tesseract

use Tesseract OCR to extract text and boxes. not recommended, as it is very slow and very bad.

```sh
 # crunch train data
 poetry run prep_docvqa_data ~/Downloads/docvqa/train train ~/Downloads/docvqa_proc_train_t1_tesseract
# crunch validation data
 poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t1_tesseract
```

#### recommended: dataset/textract

use provided dataset OCR to extract text and boxes. this was made with amazon textract and is quite decent and fast.

```sh
poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t2_dataset --ocr-engine dataset --save-ocr ~/Downloads/docvqa_proc_val_t2_ocr --procs 8
```

#### best: microsoft read ocr

use microsoft read api to extract text and boxes. this requires azure credits and is comparatively expensive, but provides the best possible ocr. with `--save-ocr` this only needs to be run once and can be used to re-encode without running ocr again.

```sh
poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t3_msread --ocr-engine microsoft --save-ocr ~/Downloads/docvqa_proc_val_t3_msread_ocr --procs 4
```

### 2. train on docvqa data

```sh
CUDA_VISIBLE_DEVICES="0" poetry run train_docvqa 'microsoft/layoutlmv3-base' ~/Downloads/docvqa_proc_val ~/Downloads/docvqa_proc_val test1 \
    --steps 100000 --batch 128 --inst-batch 8 --lr 3e-5 \
    --warmup-ratio 0.48 --save-every 100 \
    --log-wandb --project-id "llm3-docvqa-base-1"
```

### 3. (optional) segify

create segments out of bounding boxes like suggested in LayoutLMv3 and StructuralLM papers. this can give ~0.05 increase in ANLS.

this is done by simply merging nearby boxes. pass `--debug` to view visualizations one by one.

the two parameters at the end are vertical and horizontal merge distance, normalized to 1000 units. so a value of 10 width means 1% of the document's width.

```sh
poetry run segify_ocr ~/Downloads/docvqa_proc_train_t3_ocr ~/Downloads/docvqa_proc_train_t3_seg --root-dir ~/Downloads/docvqa/train 8 40
```

### 4. anls metrics

run evals to compute metrics:

```sh
poetry run budget_metrics train_output_try11_msread_segs/checkpoint-5000/ ~/Downloads/docvqa_proc_val_t7_msread/
```
