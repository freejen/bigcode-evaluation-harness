# Code Generation LM Evaluation Harness [WIP]

A framework for the evaluation of autoregressive code generation language models. 

## Overview

This project provides a unified framework to test autoregressive code generation language models.

Features:
- Any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using a code generation models trained specifically on Code such as [CodeParrot](https://huggingface.co/codeparrot/codeparrot), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
- 3 tasks implemented: [HumanEval](https://huggingface.co/datasets/openai_humaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps) and [MBPP](https://huggingface.co/datasets/mbpp).


## Setup

```bash
git clone https://github.com/bigcode-collaboration/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -r requirements.txt
```
We used `accelerate` to generate code in parallel when multiple GPUs are present. You can configure it using:

```bash
accelerate config
```
## Basic Usage

Below are some examples to evaluate a model (CodeParrot and fine-tuned GPT2 on APPS) on HumanEval and APPS benchmarks:

```bash
#to run both humaneval and apps evaluations on Codeparrot with default parameters
accelerate launch main.py \
	--model codeparrot/codeparrot \
	--tasks humaneval,apps \
	--allow_code_execution=False

#to evaluate only on some APPS samples 
accelerate launch main.py \
	--model loubnabnl/apps-1.5B-model  \
	--tasks apps \
	--level_apps introductory \
	--num_tasks_apps 10 \
    --n_samples 1 \
	--temperature 0.2 \
	--allow_code_execution=False

#to evaluate only on some MBPP samples with InCoder 1B
accelerate launch main.py \
	--model facebook/incoder-1B  \
	--prefix "<| file ext=.py |>\n" \
	--tasks mbpp \
	--num_tasks_mbpp 10 \
	--prompt_type_mbpp "incoder" \
    --n_samples 1 \
	--temperature 0.2 \
	--allow_code_execution=False
```

## Remarks
* Currenltly, we use parallel evaluation across multple GPUs using `accelerate`, this assumes that you can fit the model in one GPU. 
* Please note this evaluation harness tries to cover a wide set of models, but there could still be room for improvement based on each model, some might require different prompt engineering or post-processing of the code generations.

## Acknowledgements
This repository is inspired from [EleutherAI's LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

## To do:
- [ ] finish tests on APPS and MBPP
- [ ] add a table with some model evaluation scores