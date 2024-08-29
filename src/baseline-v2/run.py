import argparse
from tqdm import tqdm
import pandas as pd

import evaluation
from NaiveBayesClassifier import NaiveBayesClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-i', '--input_dir', type=str, default='cache/model_ckpts/baseline', help="Path to the input data")
    parser.add_argument('-o', '--output_dir', type=str, default='cache/model_output/baseline', help="Path to the output data")
    parser.add_argument('-r', '--results_dir', type=str, default='results', help="Path to the results data")
    return parser.parse_args()


def main():
    args = parse_arguments()
    results = []

    for DATASET in tqdm(['ner-disease', 'ner-gene', 'ner-pol']):

        # Load splits
        train  = pd.read_csv(f'{args.input_dir}/{DATASET}/train.csv', sep='\t')
        dev = pd.read_csv(f'{args.input_dir}/{DATASET}/dev.csv', sep='\t')
        test = pd.read_csv(f'{args.input_dir}/{DATASET}/test.csv', sep='\t')

        # Assign column names
        columns = ['sentence_number', 'word', 'label']
        train.columns = columns
        dev.columns = columns
        test.columns = columns

        # Handle NaN values in 'word' column
        train['word'] = train['word'].fillna('')
        dev['word'] = dev['word'].fillna('')
        test['word'] = test['word'].fillna('')

        # Split data into features and labels
        X_train, y_train = train['word'], train['label']
        X_dev, y_dev = dev['word'], dev['label']
        X_test, y_test = test['word'], test['label']

        # Initialize and train the model
        model = NaiveBayesClassifier()
        model.fit(X_train, y_train)

        # Evaluate model
        print('\n\n')
        dev_evaluation = model.evaluate(X_dev, y_dev, split_name='dev', dataset_name=DATASET)
        #dev_adj_evaluation = model.evaluate(X_dev, y_dev, split_name='dev', dataset_name=DATASET, adjust_pred=True)
        test_evaluation = model.evaluate(X_test, y_test, split_name='test', dataset_name=DATASET)
        #test_adj_evaluation =model.evaluate(X_test, y_test, split_name='test', dataset_name=DATASET, adjust_pred=True)
        
         
        results += [dev_evaluation, 
                    test_evaluation, 
                    #dev_adj_evaluation, 
                    #test_adj_evaluation
                    ]


    # Save results
    pd.DataFrame(results).to_json(f'{args.results_dir}/baseline-v2.jsonl', orient='records', lines=True)

    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()

