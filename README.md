# Government Climate Attention Text Mining Workflow

This repository contains the code workflow used to build climate-policy text indicators from Chinese policy and government-report corpora.

The project is organized around three tasks:

1. `train.ipynb`: train a Chinese BERT masked-language model and save the best checkpoint.
2. `UCR-vector.ipynb`: vectorize documents, sentences, and key phrases, calculate semantic similarity, and merge the indicators into panel data.
3. `Langchain-GCA.ipynb`: extract climate-related key phrases from policy text with a simple LangChain agent workflow.

The notebooks are kept as readable research records. Reusable code is placed in `src/` so the workflow can be rerun with local data paths instead of hard-coded machine paths.

## Repository Layout

```text
.
|-- src/
|   |-- train_bert.py
|   |-- vectorize_ucr.py
|   |-- extract_phrases_langchain.py
|   `-- gca_project/
|       |-- io_utils.py
|       |-- panel.py
|       |-- similarity.py
|       |-- text_utils.py
|       `-- vectorization.py
|-- data/
|   `-- README.md
|-- train.ipynb
|-- UCR-vector.ipynb
|-- Langchain-GCA.ipynb
|-- requirements.txt
|-- .env.example
`-- .gitignore
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Data

Place raw local data under `data/`. The folder is ignored by Git so private data, model checkpoints, and generated vectors are not accidentally published.

Recommended raw-data layout:

```text
data/
  train.csv
  eval.csv
  panel.csv          # optional base panel data
  policy_pdfs/
```

Generated files should go under `outputs/`. For example, extracted phrase tables, phrase vectors, similarity indicators, trained checkpoints, and merged panel data are outputs rather than raw inputs.

Expected columns can be changed from the command line. In the commands below, replace `<text_column>`, `<year_column>`, and `<region_column>` with the actual column names in your files.

## 1. Train BERT

```bash
python src/train_bert.py ^
  --train-csv data/train.csv ^
  --eval-csv data/eval.csv ^
  --text-column <text_column> ^
  --model-name bert-base-chinese ^
  --output-dir outputs/best_model ^
  --num-train-epochs 3
```

The script saves training checkpoints under `outputs/best_model` and the final reusable model under `outputs/best_model/final`.

## 2. Extract Key Phrases with LangChain

Copy `.env.example` to `.env`, then fill in the API key for your provider.

```bash
python src/extract_phrases_langchain.py build-index ^
  --pdf-dir data/policy_pdfs ^
  --vectorstore outputs/climate_policy_vectorstore
```

```bash
python src/extract_phrases_langchain.py extract ^
  --vectorstore outputs/climate_policy_vectorstore ^
  --output-xlsx outputs/policy_key_phrases.xlsx ^
  --provider qwen
```

Normalize the extracted phrase list into a table that can be vectorized:

```bash
python src/extract_phrases_langchain.py normalize-excel ^
  --input-xlsx outputs/policy_key_phrases.xlsx ^
  --output-xlsx outputs/phrases_normalized.xlsx
```

## 3. Build Vectors and Similarity Indicators

Document vectors:

```bash
python src/vectorize_ucr.py documents ^
  --input-csv data/eval.csv ^
  --text-column <text_column> ^
  --model-dir outputs/best_model/final ^
  --output-dir outputs/document_vectors
```

Sentence vectors:

```bash
python src/vectorize_ucr.py sentences ^
  --input-csv data/eval.csv ^
  --text-column <text_column> ^
  --model-dir outputs/best_model/final ^
  --output-dir outputs/sentence_vectors ^
  --group-columns <year_column> <region_column>
```

Phrase vectors:

```bash
python src/vectorize_ucr.py phrases ^
  --input-file outputs/phrases_normalized.xlsx ^
  --text-column <phrase_column> ^
  --model-dir outputs/best_model/final ^
  --output-dir outputs/phrase_vectors
```

Similarity aggregation:

```bash
python src/vectorize_ucr.py similarity ^
  --sentence-vectors outputs/sentence_vectors/vectors.npy ^
  --sentence-mapping outputs/sentence_vectors/mapping.csv ^
  --phrase-vectors outputs/phrase_vectors/vectors.npy ^
  --output-csv outputs/similarity_by_panel.csv ^
  --group-columns <year_column> <region_column>
```

If you have a base panel file, merge the indicators back into it:

```bash
python src/vectorize_ucr.py merge-panel ^
  --panel-csv data/panel.csv ^
  --similarity-csv outputs/similarity_by_panel.csv ^
  --output-csv outputs/panel_with_similarity.csv ^
  --key-columns <year_column> <region_column>
```
