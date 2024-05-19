# Utils Directory

This directory contains utility scripts used in the project.

## IMDb Dataset Preprocessing Script

`preprocess_imdb_dataset.py` processes the original IMDb dataset into three distinct splits and saves the outputs as CSV files.

### Output Directory
All output files are saved in the `../data/imdb/` directory, which will contain:
- `train.csv` (35,000 samples)
- `val.csv` (5,000 samples)
- `test.csv` (10,000 samples)

### Dataset Source
This script downloads and processes the IMDb dataset from Stanford AI Lab.
- **Dataset URL**: [IMDb Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

### Usage
To run the script, use the following command in your terminal:
```bash
python preprocess_imdb_dataset.py