from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABEL_MAPPING = {0: "negative", 1: "positive", 2: "neutral"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhoBERT for Vietnamese sentiment analysis.")
    parser.add_argument("--train-file", default="data/processed/train_new_cleaned.csv")
    parser.add_argument("--val-file", default="data/processed/val_cleaned.csv")
    parser.add_argument("--text-column", default="comments_clean")
    parser.add_argument("--label-column", default="flag")
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--output-dir", default="outputs/phobert-sentiment")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataframe(path: str, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[[text_column, label_column]].dropna()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""].copy()
    df[label_column] = df[label_column].astype(int)
    return df


def build_dataset(df: pd.DataFrame, text_column: str, label_column: str) -> Dataset:
    dataset = Dataset.from_pandas(df[[text_column, label_column]], preserve_index=False)
    return dataset.rename_column(label_column, "label")


def main() -> None:
    args = parse_args()

    train_df = load_dataframe(args.train_file, args.text_column, args.label_column)
    val_df = load_dataframe(args.val_file, args.text_column, args.label_column)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = build_dataset(train_df, args.text_column, args.label_column)
    val_dataset = build_dataset(val_df, args.text_column, args.label_column)

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_MAPPING),
        id2label=LABEL_MAPPING,
        label2id={label: idx for idx, label in LABEL_MAPPING.items()},
    )

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "weighted_f1": f1_score(labels, predictions, average="weighted"),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
