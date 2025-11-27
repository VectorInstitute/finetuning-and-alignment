# Parameter Efficient Fine-Tuning

## Overview
This directory provides demonstrations of various Parameter Efficient Fine-Tuning (PEFT) techniques.

## Contents
- **PEFT Demo** (`peft_demo.ipynb`): A demo on using PEFT techniques like LoRA and DORA, and their implementations, with functions to be completed by participants.
- **Custom Lightning Module** (`custom_lightning_module.py`): A Python module that defines custom Lightning components used in the notebook. This module is crucial for the implementation of the custom LoRA layer described in the tutorial.

## Current Demos

### PEFT Tutorial
- **Filename**: `peft_demo.ipynb`
- **Objective**: To fine-tune a model using PEFT methods, with an emphasis on practical implementations using custom LoRA layers and exploration of DORA technique.
- **Key Features**:
  - Detailed explanation and implementation of a custom LoRA layer.
  - Uses the custom LoRA layer to fine-tune a [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) model on the IMDb dataset, from the `../../data/imdb` directory.
  - Demonstrates the application of PEFT techniques using the HuggingFace [peft library](https://github.com/huggingface/peft).
  - Provides a showcase on DORA.
  - Contains incomplete functions indicated by "TODO" comments which participants are expected to implement as part of the learning process.
