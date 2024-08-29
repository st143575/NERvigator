import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
import utils

class NER:
    def __init__(self, model_name, num_labels, device='cpu', seed=42):
        self.device = device
        self.seed = seed
        self.set_seed()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
        
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
    
    @staticmethod
    def load_data(input_dir, dataset):
        train = pd.read_json(f'{input_dir}/{dataset}/train.jsonl', lines=True)
        dev = pd.read_json(f'{input_dir}/{dataset}/dev.jsonl', lines=True)
        test = pd.read_json(f'{input_dir}/{dataset}/test.jsonl', lines=True)
        return train, dev, test

    @staticmethod
    def merge_tokens_and_labels(tokens, labels):
        merged_tokens = []
        merged_labels = []
        current_token = ""
        current_label = None
            
        for token, label in zip(tokens, labels):
            if token.startswith("##"):
                current_token += token[2:]
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    merged_labels.append(current_label)
                current_token = token
                current_label = label
        
        if current_token:
            merged_tokens.append(current_token)
            merged_labels.append(current_label)
        
        final_tokens = []
        final_labels = []
        i = 0
        while i < len(merged_tokens):
            if i < len(merged_tokens) - 1 and merged_tokens[i + 1].startswith('-'):
                final_tokens.append(merged_tokens[i] + merged_tokens[i + 1])
                final_labels.append(merged_labels[i])
                i += 2
            else:
                final_tokens.append(merged_tokens[i])
                final_labels.append(merged_labels[i])
                i += 1
                
        return final_tokens, final_labels

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def prepare_data(self, sentences, labels, label2id):
        tokenized_texts_and_labels = [self.tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        input_ids = [torch.tensor(self.tokenizer.convert_tokens_to_ids(txt)) for txt in tokenized_texts]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

        tags = [torch.tensor([label2id.get(l) for l in lab]) for lab in labels]
        tags = pad_sequence(tags, batch_first=True, padding_value=label2id["PAD"])

        attention_masks = [[float(i != 0.0) for i in seq] for seq in input_ids]

        return input_ids, torch.tensor(tags), torch.tensor(attention_masks)

    def create_dataloader(self, input_ids, tags, attention_masks, batch_size):
        data = TensorDataset(input_ids, tags, attention_masks)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def train_model(self, train_dataloader, val_dataloader, optimizer, scheduler, epochs, id2label):
        for epoch in tqdm(range(epochs), desc='Epochs'):
            # Training loop
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc='Training'):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_labels, b_masks = batch

                self.model.zero_grad()

                outputs = self.model(b_input_ids, attention_mask=b_masks, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss}")

            #"""
            # Validation loop
            self.model.eval()
            eval_loss = 0
            eval_steps = 0
            predictions , true_labels = [], []
            
            for batch in tqdm(val_dataloader, desc='Validation'):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_labels, b_masks = batch

                with torch.no_grad():
                    outputs = self.model(b_input_ids, attention_mask=b_masks, labels=b_labels)
                    loss = outputs.loss
                    eval_loss += loss.item()

                    logits = outputs.logits
                    predictions.extend([list(p) for p in np.argmax(logits.detach().cpu().numpy(), axis=2)])
                    true_labels.extend(b_labels.detach().cpu().numpy().tolist())

                eval_steps += 1

            avg_eval_loss = eval_loss / eval_steps
            print(f"Validation loss: {avg_eval_loss}")
            #"""
            
        return self.model
    
    def predict(self, sentence, id2label):
        self.model.eval()
        
        # Tokenize the sentence
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        attention_mask = torch.tensor([[1] * len(input_ids[0])]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

        predicted_labels = [id2label[int(pred)] for pred in predictions[0]]

        # Merge tokens and labels
        merged_tokens, merged_labels = self.merge_tokens_and_labels(tokenized_sentence, predicted_labels)
            
        # Combine tokens with labels
        result = list(zip(merged_tokens, merged_labels))
        
        return result
    
    def evaluation_loop(self, test_data, id2label):
        self.model.eval()
        
        e_pred, e_true = [], []

        for sent, labels in tqdm(zip(test_data.sentence, test_data.tags), desc='Evaluation', total=len(test_data.sentence)):
            predictions = self.predict(" ".join(sent), id2label)
            predicted_labels = [label[1] for label in predictions]
            
            e_pred.append(utils.extract_entities(predicted_labels))
            e_true.append(utils.extract_entities(labels))

        return e_pred, e_true