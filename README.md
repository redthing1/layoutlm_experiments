# layoutlm_experiments

## finetune on docvqa

### 1. process docvqa data for LayoutLMv3

```sh
 # crunch train data
 poetry run prep_docvqa_data ~/Downloads/docvqa/train train ~/Downloads/docvqa_proc_train
# crunch validation data
 poetry run prep_docvqa_data ~/Downloads/docvqa/val val ~/Downloads/docvqa_proc_val
```

### 2. train on docvqa data

```sh
CUDA_VISIBLE_DEVICES="0" poetry run train_docvqa 'microsoft/layoutlmv3-base' ~/Downloads/docvqa_proc_train ~/Downloads/docvqa_proc_val
```
