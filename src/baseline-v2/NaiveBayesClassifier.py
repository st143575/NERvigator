import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import evaluation

class NaiveBayesClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()

    def fit(self, X, y):
        X_transformed = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X)
        return self.classifier.predict(X_transformed)
    
    def post_process_predictions(self, predictions):
        
        if predictions[0][0] == 'I':
            predictions[0] = 'B' + predictions[0][:1]
        
        new_preds = [predictions[0]]

        for prev, curr in zip(predictions[:-1], predictions[1:]):
             
            if prev == 'O' and curr[0] == 'I':
                curr = 'B' + curr[1:]
            elif prev[0] == 'B' and curr[0] == 'B':
                curr = 'I'+ curr[1:]

            new_preds.append(curr)

        return new_preds

    def extract_entities(self, preds):
        """
        Extract entities (span)
        """

        entities = []
        current_entity = None
        start_index = None

        for i, label in enumerate(preds):
            if label.startswith('B-'):
                
                # New entity starts
                if current_entity:
                    current_entity['end'] = i - 1
                    entities.append(current_entity)
                current_entity = {'type': label[2:], 'start': i, 'end': None}
            
            elif label.startswith('I-'):
                # Continuing the current entity
                if current_entity:
                    if start_index is None:
                        start_index = i - 1
                    current_entity['end'] = i

            elif label == 'O':
                # End of entity
                if current_entity:
                    current_entity['end'] = i - 1
                    entities.append(current_entity)
                    current_entity = None
                    start_index = None

        # If there's an entity still being processed
        if current_entity:
            current_entity['end'] = len(preds) - 1
            entities.append(current_entity)

        return entities

    def evaluate(self, X_test, y_test, split_name='test', dataset_name='ner', adjust_pred=False, verbose=False):
        """
        Evaluation on a span level
        """
        y_pred = self.predict(X_test)
        
        # Post process predictions
        if adjust_pred == True:
            y_pred = self.post_process_predictions(y_pred)

        e_pred = self.extract_entities(y_pred)
        e_true = self.extract_entities(y_test)

        

        #exact_match_score = evaluation.exact_match(e_pred, e_true)
        precision = evaluation.precision(e_pred, e_true)
        recall = evaluation.recall(e_pred, e_true)
        f1_score = evaluation.f1_score(e_pred, e_true)

        evaluation_dict = {
            'model': 'nb-sci-ner_adj' if adjust_pred == True else 'nb-sci-ner',
            'split': split_name,
            'dataset': dataset_name,
            #'exact_match_score': exact_match_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        if verbose == True:
            print('---', split_name, dataset_name, '---')
            print(f'Exact Match Score: {exact_match_score:.2f}')
            print(f'Precision: {precision:.2f}')
            print(f'Recall: {recall:.2f}')
            print(f'F1 Score: {f1_score:.2f}', '\n')

        return evaluation_dict

