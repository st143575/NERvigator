import os, argparse
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import env, utils
from NER import NER

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-i', '--input_dir', type=str, default='cache/model_ckpts/nli', help="Path to the input data")
    parser.add_argument('-o', '--output_dir', type=str, default='cache/model_output/nli', help="Path to the output data")
    parser.add_argument('-r', '--results_dir', type=str, default='results', help="Path to the results data")
    parser.add_argument('-c', '--cache_dir', type=str, default='cache/model_ckpts/nli', help="Path to the model checkpoints")
    parser.add_argument('-d', '--device', type=str, default='cpu', help="Torch Device ('cpu' or 'gpu')'")
    parser.add_argument('-m', '--model_name', type=str, default='roberta-base', help="Model name")
    parser.add_argument('-e', '--epochs', type=int, default=0, help="Number of epochs (default: zero-shot)")
    return parser.parse_args()


def trainer(model, train_dataloader, val_dataloader, id2label, epochs=None):
    # Fine-tuning setup
    optimizer = AdamW(model.model.parameters(), lr=3e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the model (if epochs=None do zeroshot)
    if epochs:
        model.train_model(train_dataloader, val_dataloader, optimizer, scheduler, epochs, id2label)


def main():
    args = parse_arguments()

    if args.device:
        device = args.device
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(env.GPU)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for DATASET in tqdm(env.DATASETS, desc=f'Datasets', total=len(env.DATASETS)):

        # Load data
        print("Load splits...")  
        train_data, dev_data, test_data = NER.load_data(args.input_dir, DATASET)

        # Define labels and their mappings
        id2label = env.id2label[DATASET]
        labels = list(id2label.values())
        label2id = {label: idx for idx, label in enumerate(labels)}

        # Initialize model
        ner = NER(args.model_name, num_labels=len(labels), device=device)

        # Prepare data
        print('Prepare data...')
        train_input_ids, train_tags, train_masks = ner.prepare_data(train_data.sentence, train_data.tags, label2id)
        val_input_ids, val_tags, val_masks = ner.prepare_data(dev_data.sentence, dev_data.tags, label2id)

        train_dataloader = ner.create_dataloader(train_input_ids, train_tags, train_masks, batch_size=32)
        val_dataloader = ner.create_dataloader(val_input_ids, val_tags, val_masks, batch_size=32)

        
        # Fine-tune
        if args.epochs > 0: # if EPOCHS == 0 --> zero-shot
            print(f'Train model for {args.epochs} epochs...')
            trainer(ner, train_dataloader, val_dataloader, id2label, epochs=args.epochs)

        # Evaluate
        print(f'Evaluate "epochs-{args.epochs}" model...')
        e_pred, e_true = ner.evaluation_loop(test_data, id2label)

        # Save predictions
        print(f'Save "epochs-{args.epochs}" outputs...')
        utils.save_entities_to_json(e_pred, e_true, f'{args.output_dir}/{DATASET}/{args.model_name}_epochs-{args.epochs}_test-entities.json')



if __name__ == '__main__':
    main()
        