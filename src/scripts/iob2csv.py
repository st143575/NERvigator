import os, argparse
import pandas as pd
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-i', '--input_dir', type=str, default='../../datasets', help="Path to the input data")
    parser.add_argument('-o', '--output_dir', type=str, default='../../cache/model_ckpt', help="Path to the output data")
    return parser.parse_args()


def load_iob_to_df(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sentence_number = 0
    data = {
        'sentence_number': [],
        'word': [],
        'label': []
    }
    
    for line in lines:
        if line.strip():
            word, label = line.split('\t|')
            
            # Ignore -DOCSTART- 
            if word == '-DOCSTART-':
                sentence_number -=1
                continue

            word = word.strip()
            label = label.strip()
            
            # Update
            data['sentence_number'].append(sentence_number)
            data['word'].append(word)
            data['label'].append(label)
        else:
            sentence_number += 1
    
    df = pd.DataFrame(data)
    return df


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")

    for subdir, dirs, files in os.walk(input_dir):
        dataset_name = Path(subdir).name

        for file in files:
            if file.endswith('.iob'):
                data_split = file.split('.')[0]

                print(f"Start preprocessing dataset {dataset_name}_{data_split}...")
                dataset = load_iob_to_df(f"{subdir}/{file}")

                # Save in cache
                dataset.to_csv(f'{output_dir}/{dataset_name}/{data_split}.csv', sep='\t', index=False, header=None)

            


if __name__ == "__main__":
    main()