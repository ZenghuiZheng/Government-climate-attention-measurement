import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from gca_project.text_utils import ensure_columns, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese BERT masked-language model.")
    parser.add_argument("--train-csv", required=True, help="Training CSV path.")
    parser.add_argument("--eval-csv", required=True, help="Validation CSV path.")
    parser.add_argument("--text-column", default="报告全文", help="Text column name.")
    parser.add_argument("--model-name", default="bert-base-chinese", help="HF model name or local model path.")
    parser.add_argument("--output-dir", default="outputs/best_model", help="Checkpoint and best-model directory.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--logging-steps", type=int, default=1000)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-tokenized-dir", default=None, help="Optional directory for tokenized datasets.")
    return parser.parse_args()


def load_dataset(csv_path: str, text_column: str) -> Dataset:
    df = pd.read_csv(csv_path)
    ensure_columns(df.columns, [text_column])
    df[text_column] = df[text_column].fillna("").map(normalize_text)
    df = df[df[text_column] != ""].reset_index(drop=True)
    return Dataset.from_pandas(df[[text_column]], preserve_index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)

    train_dataset = load_dataset(args.train_csv, args.text_column)
    eval_dataset = load_dataset(args.eval_csv, args.text_column)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_attention_mask=True,
        )

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

    if args.save_tokenized_dir:
        tokenized_root = Path(args.save_tokenized_dir)
        tokenized_train.save_to_disk(str(tokenized_root / "tokenized_train"))
        tokenized_eval.save_to_disk(str(tokenized_root / "tokenized_eval"))

    del train_dataset, eval_dataset
    gc.collect()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16 and torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=300,
        weight_decay=0.01,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()
