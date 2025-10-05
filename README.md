# Jigsaw Agile Community Rules Classification

This project contains code and experiments for the [Jigsaw Agile Community Rules Kaggle competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview).

## Overview

The goal of this competition is to build a model that can determine whether a Reddit comment violates a specific community rule. This involves natural language processing and text classification.

The approach taken in this project involves:
- Fine-tuning various Large Language Models (LLMs) like Qwen on the competition data.
- Using techniques like LoRA (Low-Rank Adaptation) for efficient fine-tuning.
- Performing semantic search to find similar examples.
- Ensembling the predictions from multiple models to generate the final submission.

## Project Structure

The repository is organized as follows:

```
├── data/              # Raw and processed data
├── models/            # Trained model artifacts
├── notebooks/         # Jupyter notebooks for experimentation
├── src/               # Source code for data processing, training, and inference
├── submissions/       # Generated submission files
├── GEMINI.md          # Detailed context for the Gemini CLI
└── README.md          # This file
```

## Getting Started

### Prerequisites

You will need Python and the dependencies listed below. It is recommended to use a virtual environment.

### Dependencies

A `requirements.txt` file is not yet available. The main dependencies are:
- `pandas`
- `datasets`
- `trl`
- `peft`
- `transformers`
- `vllm`
- `sentence-transformers`
- `cleantext`
- `torch`
- `scikit-learn`

### Running the Code

The main workflow is contained within `notebooks/notebook.ipynb`. To run the project:

1.  **Download the data** from the Kaggle competition page and place the CSV files in the `data/` directory.
2.  **Open and run the `notebooks/notebook.ipynb` notebook.** This notebook is set up to be run in a Kaggle environment but can be adapted for local execution. It handles data preparation, training, inference, and generates the final `submission.csv` file.
