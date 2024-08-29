import json
import evaluate
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import List
# from evaluate import load


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the natural language generation for NER (0-shot).')
    parser.add_argument('-i', '--input_dir', type=str, default="/mount/studenten-temp1/users/cs/SS2024/cl-team-lab-ner/src/advanced_approach/nlg/output")
    parser.add_argument('-dsn', '--dataset_name', type=str, required=True, help="Name of the dataset, should be one of 'ner-disease', 'ner-gene', 'ner-pol'.")
    parser.add_argument('-o', '--output_dir', type=str, default="/mount/studenten-temp1/users/cs/SS2024/cl-team-lab-ner/src/advanced_approach/nlg/evaluation/output")
    parser.add_argument('-m', '--model', type=str, default="gpt-4o")
    return parser.parse_args()


def compute_metrics(metrics, y_trues: List[List[str]], y_preds: List[List[str]]):
    results = metrics.compute(predictions=y_preds, references=y_trues, mode="strict")
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load the seqeval framework.
    seqeval = evaluate.load("seqeval")
    # seqeval = load("seqeval")
    
    # Load postprocessed predictions.
    ner_pred_labels_df = pd.read_json(f"{input_dir}/{args.dataset_name}-{args.model}-0-shot-labels.jsonl", lines=True)

    # Compute metrics.
    ner_metrics = compute_metrics(seqeval, ner_pred_labels_df["tags"], ner_pred_labels_df["tags_pred"])

    # Write metrics to file.
    with open(f"{output_dir}/{args.dataset_name}-{args.model}-0-shot-metrics.json", "w") as f:
        json.dump(ner_metrics, f)

if __name__ == "__main__":
    main()