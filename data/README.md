# Data Directory

This directory contains datasets used in the repository, organized as follows:

## IMDb Dataset

The `imdb` directory includes data that has been processed and split into training, testing, and evaluation sets. 

### Source
- **Original Dataset:** [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
  - This dataset was created by the Stanford AI Lab and contains movie reviews along with sentiment polarity labels.

### Files
- `train.csv` (35,000 samples)
- `val.csv` (5,000 samples)
- `test.csv` (10,000 samples)

### Generation
These files were generated using the script located at `utils/preprocess_imdb_dataset.py`. The script processes the original IMDb dataset to create three distinct splits: train, test, and eval.

## Debiased Profanity Check Dataset

This file, `debiased_profanity_check_with_keywords.csv`, contains data related to profanity and bias checks, with specific keywords highlighted for analysis.

### Source
- **Dataset:** [Bias-DeBiased](https://huggingface.co/datasets/newsmediabias/Bias-DeBiased)
  - This dataset is part of efforts to understand and mitigate bias in media texts, hosted on Hugging Face.

### Reference Paper
- [Exploring the Detection of Media Bias via Machine Learning](https://arxiv.org/abs/2404.01399)
  - This paper discusses methodologies for detecting and debiasing media bias using advanced machine learning techniques.


## Citation
If you use the data provided in this directory, please cite the appropriate sources as follows:

```bibtex

@misc{raza2023newsmediabias,
  Author     = {Shaina Raza},
  title     = {News Media Bias},
  year      = {2023},
  url       = {https://huggingface.co/newsmediabias},
}
