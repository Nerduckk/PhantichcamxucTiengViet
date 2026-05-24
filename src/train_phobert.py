from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
DISPLAY_LABELS = ["Negative", "Neutral", "Positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhoBERT for Vietnamese sentiment analysis.")
    parser.add_argument("--train-file", default="data/processed/train_new_cleaned.csv")
    parser.add_argument("--val-file", default="data/processed/val_cleaned.csv")
    parser.add_argument("--test-file", default="data/processed/test_cleaned.csv")
    parser.add_argument("--text-column", default="comments_clean")
    parser.add_argument("--label-column", default="flag")
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--output-dir", default="outputs/phobert-sentiment")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
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


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    }


def main() -> None:
    args = parse_args()

    train_df = load_dataframe(args.train_file, args.text_column, args.label_column)
    val_df = load_dataframe(args.val_file, args.text_column, args.label_column)
    test_df = load_dataframe(args.test_file, args.text_column, args.label_column)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [str(text).strip() for text in batch[args.text_column]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = build_dataset(train_df, args.text_column, args.label_column).map(tokenize_batch, batched=True)
    val_dataset = build_dataset(val_df, args.text_column, args.label_column).map(tokenize_batch, batched=True)
    test_dataset = build_dataset(test_df, args.text_column, args.label_column).map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_MAPPING),
        id2label=LABEL_MAPPING,
        label2id={label: idx for idx, label in LABEL_MAPPING.items()},
        ignore_mismatched_sizes=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp16 = torch.cuda.is_available() and not args.no_fp16
    if args.fp16:
        fp16 = True

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.save_strategy != "no",
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        fp16=fp16,
        save_only_model=args.save_only_model,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    test_output = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_output.predictions, axis=1)
    test_labels = np.array(test_dataset["label"])

    test_metrics = {
        "accuracy": accuracy_score(test_labels, test_predictions),
        "macro_f1": f1_score(test_labels, test_predictions, average="macro"),
        "weighted_f1": f1_score(test_labels, test_predictions, average="weighted"),
    }
    test_report = classification_report(
        test_labels,
        test_predictions,
        target_names=DISPLAY_LABELS,
        digits=4,
        output_dict=True,
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "validation": val_metrics,
                "test": test_metrics,
                "test_report": test_report,
                "config": vars(args),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Validation metrics:")
    for key, value in val_metrics.items():
        print(f"{key}: {value}")

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
