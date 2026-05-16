from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import pipeline


LABEL_MAPPING = {"negative": 0, "positive": 1, "neutral": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PhoBERT sentiment model.")
    parser.add_argument("--model-dir", default="outputs/phobert-sentiment")
    parser.add_argument("--test-file", default="data/processed/test_cleaned.csv")
    parser.add_argument("--text-column", default="comments_clean")
    parser.add_argument("--label-column", default="flag")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.test_file)[[args.text_column, args.label_column]].dropna()
    df[args.text_column] = df[args.text_column].astype(str).str.strip()
    df = df[df[args.text_column] != ""].copy()
    df[args.label_column] = df[args.label_column].astype(int)

    classifier = pipeline(
        task="text-classification",
        model=args.model_dir,
        tokenizer=args.model_dir,
        batch_size=args.batch_size,
        truncation=True,
    )

    predictions = classifier(df[args.text_column].tolist())
    predicted_labels = np.array([LABEL_MAPPING[item["label"]] for item in predictions])
    true_labels = df[args.label_column].to_numpy()

    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print(f"Macro F1: {f1_score(true_labels, predicted_labels, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(true_labels, predicted_labels, average='weighted'):.4f}")
    print("\nClassification report:")
    print(classification_report(true_labels, predicted_labels, digits=4))


if __name__ == "__main__":
    main()
