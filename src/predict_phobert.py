from __future__ import annotations

import argparse

from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-text prediction with PhoBERT.")
    parser.add_argument("--model-dir", default="outputs/phobert-sentiment")
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classifier = pipeline(
        task="text-classification",
        model=args.model_dir,
        tokenizer=args.model_dir,
        truncation=True,
    )
    result = classifier(args.text)[0]
    print(f"label={result['label']}")
    print(f"score={result['score']:.4f}")


if __name__ == "__main__":
    main()
