# FIZZ: Factual Inconsistency Detection by Zoom-in Summary and Zoom-out Document

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2305.14251-b31b1b.svg)](https://arxiv.org/abs/2404.11184)

This repository contains the code for EMNLP2024 paper: FIZZ: Factual Inconsistency Detection by Zoom-in Summary and Zoom-out Document

## Running
```bash
python fizz/fizz_score.py
```

**Optional flags**:
- `--input_path`: Directory for input data path. We recommend using `.csv` file. `data/aggre_fact_sota.csv` by default.
- `--output_path`: Directory for output data path. We recommend including `.csv` format. `data/output.csv` by default.
- `--doc_label`: Name of the column for the source document. `doc` by default.
- `--summary_label`: Name of the column for the summary. `summary` by default.
- `--atomic_facts_column`: Name of the column to add the generated atomic fact. `atomic_facts` by default.
- `--score_column`: Name of the column to add the computed score. `FIZZ_score` by default.
- `--model_name`: Name of the LLM for atomic fact decomposition. `orca2` by default. There are `zephyr` and `mistral` choices.
- `--granularity`: Granularity choice for computing FAIRY score. `3G` by default.
