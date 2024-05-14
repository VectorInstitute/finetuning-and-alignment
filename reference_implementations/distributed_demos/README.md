# Distributed LLM Training Overview

## Introduction
Welcome to the distributed training module of the Finetuning and Alignment Bootcamp. This folder contains a collection of code samples designed to enhance your understanding of distributed training and computing. Here, we will explore Fully Sharded Data Parallel (FSDP) and Distributed Data Parallel (DDP) training.

## Prerequisites
Before you dive into the materials, if you are running this on the Vector cluster, make sure to first activate the environment by doing `source /projects/fta_bootcamp/envs/finetune_demo/bin/activate`.

## Code
This section includes code files that demonstrate:
- **dist_demo.py**: This is the main file where distributed training (both FSDP and DDP) takes place. Please read through this and familiarize yourself with the full script.
- **data.py**: This script handles the data preprocessing, distributing data to different workers (GPUs), and creating the data loaders.
- **utils.py**: This script has a few miscellaneous functions for distributed settings, and loading the model and tokenizer.

## Resources
For further reading and additional studies, consider the following resources:
- [Getting Started with Fully Sharded Data Parallel(FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## Getting Started
The main file to run is `dist_demo.py`. The first few lines of the `main` function has parameters that you can change (lines 214-220). Please refrain from changing the model or dataset as the data preprocessing assumes we are using the preset model and dataset. To launch the file, run
```bash
torchrun --nproc-per-node=NUM_GPUS dist_demo.py
```
For example, if you had 2 GPUs, you would run `torchrun --nproc-per-node=2 dist_demo.py`.