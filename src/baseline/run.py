import argparse
from tqdm import tqdm
import pandas as pd
import trainer
from RandomNER import RandomNER
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
        train  = pd.read_json(f'{args.input_dir}/{DATASET}/train.jsonl', lines=True)
        dev = pd.read_json(f'{args.input_dir}/{DATASET}/dev.jsonl', lines=True)
        test = pd.read_json(f'{args.input_dir}/{DATASET}/test.jsonl', lines=True)

        # Load word vectors
        word_vectors = pd.read_json(f'{args.input_dir}/{DATASET}/train_tfidf.jsonl', lines=True)
        print(word_vectors.head())

        # Serialize labels
        labels_map = trainer.create_labels_map(train.tags)
        inverted_labels_map = {value: key for key, value in labels_map.items()}
        classes = list(labels_map.values())
        print('Label map: ', labels_map)


        # Map labels to classes
        train['classes'] = train.tags.map(lambda tags: [labels_map[label] for label in tags])
        dev['classes'] = dev.tags.map(lambda tags: [labels_map[label] for label in tags])
        test['classes'] = dev.tags.map(lambda tags: [labels_map[label] for label in tags])
        
        # Init models
        random_ner = RandomNER(classes)
        nb_ner = NaiveBayesClassifier(classes)

        # Train models
        trainer.training_loop(nb_ner, train)

        # Make predictions
        dev_random_preds = trainer.prediction_loop(random_ner, dev.sentence, inverted_labels_map)
        test_random_preds = trainer.prediction_loop(random_ner, test.sentence, inverted_labels_map)

        dev_nb_preds = trainer.prediction_loop(nb_ner, dev.sentence, inverted_labels_map)
        test_nb_preds = trainer.prediction_loop(nb_ner, test.sentence, inverted_labels_map)

        # Cache predictions
        dev_random = dev.copy()
        dev_random['tags'] = dev_random_preds
        dev_random.to_json(f'{args.output_dir}/{DATASET}/dev-random.jsonl', orient='records', lines=True)
        
        test_random = test.copy()
        test_random['tags'] = test_random_preds
        test_random.to_json(f'{args.output_dir}/{DATASET}/test-random.jsonl', orient='records', lines=True)

        dev_nb = dev.copy()
        dev_nb['tags'] = dev_nb_preds
        dev_nb.to_json(f'{args.output_dir}/{DATASET}/dev-nb.jsonl', orient='records', lines=True)

        test_nb = test.copy()
        test_nb['tags'] = test_nb_preds
        test_nb.to_json(f'{args.output_dir}/{DATASET}/test-nb.jsonl', orient='records', lines=True)

        # Evaluation
        dev_eval_random_ner = trainer.evaluation_loop(dev.tags, dev_random.tags)
        dev_eval_random_ner['model'] = 'random-ner'
        dev_eval_random_ner['dataset'] = f'dev_{DATASET}'
        results.append(dev_eval_random_ner)

        test_eval_random_ner = trainer.evaluation_loop(test.tags, test_random.tags)
        test_eval_random_ner['model'] = 'random-ner'
        test_eval_random_ner['dataset'] = f'test_{DATASET}'
        results.append(test_eval_random_ner)
        
        dev_eval_nb_ner = trainer.evaluation_loop(dev.tags, dev_nb.tags)
        dev_eval_nb_ner['model'] = 'nb-ner'
        dev_eval_nb_ner['dataset'] = f'dev_{DATASET}'
        results.append(dev_eval_nb_ner)

        test_eval_nb_ner = trainer.evaluation_loop(dev.tags, dev_nb.tags)
        test_eval_nb_ner['model'] = 'nb-ner'
        test_eval_nb_ner['dataset'] = f'test_{DATASET}'
        results.append(test_eval_nb_ner)

    # Save results
    pd.DataFrame(results).to_json(f'{args.results_dir}/baseline.jsonl', orient='records', lines=True)

        



if __name__ == "__main__":
    main()