import pandas as pd
from pathlib import Path
import os, argparse
from tqdm import tqdm

iob2idx = {
    "ner-disease": {
        "O": 0, 
        "B-DISEASE": 1,
        "I-DISEASE": 2
    },

    "ner-gene": {
        "O": 0, 
        "B-PROTEIN": 1,
        "I-PROTEIN": 2
    },

    "ner-pol": {
        "O": 0, 
        "B-PER": 1, 
        "I-PER": 2, 
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8
    }
}

PROMPT_TEMPLATE = 'For the sentence {sentence} Predict entity of "{token}".'
PROMPT_TEMPLATE_WITH_GOLD = 'For the sentence {sentence} Predict entity of "{token}": {label}'

#TEMPLATE = """{sentence} {token}: {label}""".strip()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-i', '--input_dir', type=str, default='../../cache/model_ckpts/nli/ner-disease', help="Path to the input data")
    parser.add_argument('-o', '--output_dir', type=str, default='../../cache/model_ckpt', help="Path to the output data")
    return parser.parse_args()



def generate_promts(sentence: list[str], prompt_template: str):
    prompts = []
    for token in sentence:
        prompt = prompt_template.format(sentence=" ".join(sentence), token=token)
        prompts.append(prompt)

    return prompts



def main():
    args = parse_arguments()

    for subdir, dirs, files in os.walk(args.input_dir):
            dataset_name = Path(subdir).name

            for file in files:
                fpath = f"{subdir}/{file}"
                data_split = file.split('.')[0]
                
                if file.endswith('.jsonl'):
                    data = pd.read_json(fpath, orient='records', lines=True)
                    prompts_df = pd.DataFrame()
                    
                    print(f"Generate prompts for {dataset_name}_prompts_{data_split}")

                    for _, row in tqdm(data.iterrows()):
                        sent, labels = row.sentence, row.tags
                        prompts = generate_promts(sent, PROMPT_TEMPLATE)
                        
                        for token, label, prompt in zip(sent, labels, prompts):
                            
                            new_row = pd.DataFrame({'doc_id': [row.doc_id], 'sent_id': [row.sent_id], 'token': [token], 'prompt': [prompt], 'label': [label]})
                            prompts_df = pd.concat([prompts_df, new_row], ignore_index=True)

                    output_file = f"{args.output_dir}/{dataset_name}/prompts_{data_split}.jsonl"
                    prompts_df.to_json(output_file, orient='records', lines=True)
                    print(f"Generation of prompts {dataset_name} prompts_{data_split} done. Output saved to {output_file}\n")
                    
    

if __name__ == "__main__":
    main()