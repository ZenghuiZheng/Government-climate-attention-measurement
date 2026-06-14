from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, default_data_collator

from gca_project.io_utils import read_table, save_vectors
from gca_project.text_utils import ensure_columns, normalize_text, split_sentences


@dataclass
class VectorizationConfig:
    text_column: str
    model_dir: str
    output_dir: str
    max_length: int = 256
    batch_size: int = 64
    num_workers: int = 4


def load_model(model_dir: str) -> tuple[BertTokenizer, BertModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def embed_texts(
    texts: list[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    device: torch.device,
    max_length: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=num_workers,
    )

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            cls_vectors = outputs.last_hidden_state[:, 0, :]
            vectors.append(cls_vectors.cpu().numpy().astype(np.float32))

    if not vectors:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(vectors)


def vectorize_documents(input_csv: str, group_columns: list[str], config: VectorizationConfig) -> None:
    df = pd.read_csv(input_csv)
    ensure_columns(df.columns, [config.text_column, *group_columns])
    df[config.text_column] = df[config.text_column].fillna("").map(normalize_text)
    df = df[df[config.text_column] != ""].reset_index(drop=True)

    vectors = _embed_from_config(df[config.text_column].tolist(), config)
    mapping_columns = [*group_columns, config.text_column]
    save_vectors(config.output_dir, vectors, df[mapping_columns])


def vectorize_sentences(
    input_csv: str,
    group_columns: list[str],
    min_sentence_length: int,
    config: VectorizationConfig,
) -> None:
    df = pd.read_csv(input_csv)
    ensure_columns(df.columns, [config.text_column, *group_columns])

    records = []
    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Splitting sentences"):
        for sentence_index, sentence in enumerate(split_sentences(row[config.text_column], min_sentence_length)):
            record = {"source_row": row_index, "sentence_index": sentence_index, "text": sentence}
            for column in group_columns:
                record[column] = row[column]
            records.append(record)

    mapping = pd.DataFrame(records)
    if mapping.empty:
        raise ValueError("No sentences were generated. Check the text column and sentence length settings.")

    vectors = _embed_from_config(mapping["text"].tolist(), config)
    save_vectors(config.output_dir, vectors, mapping)


def vectorize_phrases(input_file: str, sheet_name: str | None, config: VectorizationConfig) -> None:
    df = read_table(input_file, sheet_name)
    ensure_columns(df.columns, [config.text_column])
    df[config.text_column] = df[config.text_column].fillna("").map(normalize_text)
    df = df[df[config.text_column] != ""].drop_duplicates(config.text_column).reset_index(drop=True)

    vectors = _embed_from_config(df[config.text_column].tolist(), config)
    save_vectors(config.output_dir, vectors, df[[config.text_column]])


def _embed_from_config(texts: list[str], config: VectorizationConfig) -> np.ndarray:
    tokenizer, model, device = load_model(config.model_dir)
    return embed_texts(
        texts,
        tokenizer,
        model,
        device,
        config.max_length,
        config.batch_size,
        config.num_workers,
    )
