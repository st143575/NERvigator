import argparse
from tqdm import tqdm
import pandas as pd
import json
import env, utils, csv
import seaborn as sns
import matplotlib as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('-i', '--input_dir', type=str, default='cache/model_output/nli', help="Path to the input data")
    parser.add_argument('-o', '--output_dir', type=str, default='results', help="Path to the output data")
    parser.add_argument('-m', '--model_name', type=str, default='roberta-base', help="Model name")
    parser.add_argument('-e', '--epochs', type=int, default=0, help="Number of epochs (default: zero-shot)")
    return parser.parse_args()

def exact_match(e_pred: list[dict], e_true: list[dict]) -> int:
    if not e_pred and not e_true:
        return 1
    elif not e_pred or not e_true:
        return 0
    else:
        return sum(e in e_true for e in e_pred)


def precision(e_pred: list[dict], e_true: list[dict]) -> float:
    true_positives = sum(e in e_true for e in e_pred) # exact match
    predicted_positives = len(e_pred)

    if predicted_positives == 0:
        return 1.0 if true_positives == 0 else 0.0

    return true_positives / predicted_positives



def recall(e_pred: list[dict], e_true: list[dict]) -> float:
    true_positives = sum(e in e_true for e in e_pred) # exact match
    actual_positives = len(e_true)
    
    if actual_positives == 0:
        return 1.0 if true_positives == 0 else 0.0
    
    return true_positives / actual_positives

def f1_score(e_pred: list[dict], e_true: list[dict]) -> float:

    prec = precision(e_pred, e_true)
    rec = recall(e_pred, e_true)

    if (prec + rec) == 0:
        return 0.0
    else:
        return 2*((prec * rec) / (prec + rec))

"""
def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
"""

    

def main():
    args = parse_arguments()
    results = []

    # Read existing results from nli.jsonl
    existing_results_path = f'{args.output_dir}/nli.jsonl'
    try:
        with open(existing_results_path, 'r') as file:
            existing_results = [json.loads(line) for line in file]
    except FileNotFoundError:
        existing_results = []

    # Deals with duplicates
    existing_results_set = {
        (result['model'], result['epochs'], result['split'], result['dataset'])
        for result in existing_results
    }

    for DATASET in tqdm(env.DATASETS, desc='Datasets', total=len(env.DATASETS)):

        input_fname=f'{args.input_dir}/{DATASET}/{args.model_name}_epochs-{args.epochs}_test-entities.json'

        e_pred, e_true = utils.read_entities_from_json(input_fname)
        
            
        print('\n', exact_match(e_pred, e_true), 'exact matches out of', len(e_pred), 'predicted entities and', len(e_true), 'true entities.\n')

        # Evaluate model
        prec, rec, f1 = [], [], []

        for sent_e_pred, sent_e_true in zip(e_pred, e_true):
            # Evaluate model per sentence
            prec.append(precision(sent_e_pred, sent_e_true))  # precision and exact match score are the same!
            rec.append(recall(sent_e_pred, sent_e_true))
            f1.append(f1_score(sent_e_pred, sent_e_true))

        evaluation_dict = {
            'model': args.model_name,
            'epochs': int(args.epochs),
            'split': 'test',
            'dataset': DATASET,
            'precision': sum(prec) / len(prec),
            'recall': sum(rec) / len(rec),
            'f1_score': sum(f1) / len(f1)
        }

        # Deals with duplicates
        if (evaluation_dict['model'], evaluation_dict['epochs'], evaluation_dict['split'], evaluation_dict['dataset']) not in existing_results_set:
            results.append(evaluation_dict)

    # Combine new results with existing results
    combined_results = existing_results + results

    # Save combined results
    with open(existing_results_path, 'w') as file:
        for result in combined_results:
            file.write(json.dumps(result) + '\n')

    # Optionally, print the DataFrame for verification
    print(pd.DataFrame(combined_results))

    



if __name__ == '__main__':
     main()