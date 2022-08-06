# layoutlm_experiments

a work in progress project, to replicate LayoutLMv3's performance on DocVQA.


## 1. process docvqa data for LayoutLMv3

### not recommended: tesseract

use Tesseract OCR to extract text and boxes. not recommended, as it is very slow and very bad.

```sh
 # crunch train data
 poetry run prep_docvqa_data ~/Downloads/docvqa/train train ~/Downloads/docvqa_proc_train_t1_tesseract
# crunch validation data
 poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t1_tesseract
```

### recommended: dataset/textract

use provided dataset OCR to extract text and boxes. this was made with amazon textract and is quite decent and fast.

```sh
poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t2_dataset --ocr-engine dataset --save-ocr ~/Downloads/docvqa_proc_val_t2_ocr --procs 8
```

### best: microsoft read ocr

use microsoft read api to extract text and boxes. this requires azure credits and is comparatively expensive, but provides the best possible ocr. with `--save-ocr` this only needs to be run once and can be used to re-encode without running ocr again.

```sh
poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t3_msread --ocr-engine microsoft --save-ocr ~/Downloads/docvqa_proc_val_t3_msread_ocr --procs 4
```

### optional: segify

create segments out of bounding boxes like suggested in LayoutLMv3 and StructuralLM papers. this can give ~0.05 increase in ANLS.

this is done by simply merging nearby boxes. pass `--debug` to view visualizations one by one.

the two parameters at the end are vertical and horizontal merge distance, normalized to 1000 units. so a value of 10 width means 1% of the document's width.

segify needs to be run on the saved ocr output of the last step.

```sh
poetry run segify_ocr ~/Downloads/docvqa_proc_train_t3_ocr ~/Downloads/docvqa_proc_train_t3_seg --root-dir ~/Downloads/docvqa/train 8 40
```

after segify is finished, re-encode the dataset:

```sh
poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t7_msread --resume-from-ocr ~/Downloads/docvqa_proc_val_t3_msread_seg --procs 8
```

## 2. train on docvqa data

```sh
CUDA_VISIBLE_DEVICES="0" poetry run train_docvqa 'microsoft/layoutlmv3-base' ~/Downloads/docvqa_proc_val ~/Downloads/docvqa_proc_val test1 \
    --steps 100000 --batch 128 --inst-batch 8 --lr 3e-5 \
    --warmup-ratio 0.48 --save-every 100 \
    --log-wandb --project-id "llm3-docvqa-base-1"
```

## 4. anls metrics

run evals to compute metrics:

```sh
poetry run budget_metrics train_output_try11_msread_segs/checkpoint-5000/ ~/Downloads/docvqa_proc_val_t7_msread/
```

## alternate approach: LayoutLMv3Bart

LayoutLMv3 encoder with BART decoder

stack a pretrained decoder onto a pretrained layoutlmv3 and finetune together for seq2seq, following [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)

1. fuse together layoutlmv3 encoder with bart decoder

```sh
poetry run fuse_encdec 'microsoft/layoutlmv3-base' 'facebook/bart-base' --save-to ~/Downloads/PT_LLMv3_BART_FUSE_T1 --test-input ~/Downloads/dsconst/freebirds.jpg
```

2. preprocess for seq2seq
```sh
poetry run prep_docvqa_seq2seq ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val_t6 --tiny-subset --ocr-engine dataset --decoder-model 'facebook/bart-base'
```

3. train for seq2seq
```sh
poetry run train_docvqa_seq2seq ~/Downloads/PT_LLMv3_BART_FUSE_T1/ ~/Downloads/docvqa_proc_train_t8_seq2seq_msread ~/Downloads/docvqa_proc_val_t8_seq2seq_msread try17_msr2s_s2s --steps 100000 --batch 32 --inst-batch 4 --lr 5e-6 --warmup-ratio 0.048 --save-every 1000
```
