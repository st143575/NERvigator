import pandas as pd
import numpy as np
from tqdm import tqdm



def create_labels_map(labeled_sentences: list):
        cls = 0
        labels_map = {'O': cls}
        for labeled_sent in labeled_sentences:
            for label in labeled_sent:
                if label not in labels_map:
                    cls += 1
                    labels_map[label] = cls
        
        return labels_map

def prediction_loop(model, sentences, inverted_labels_map, word_vectors=None):

    predictions = []
    for sentence in tqdm(sentences):
        
        if word_vectors is not None:
            X_pred = pd.DataFrame([word_vectors.get(word, np.zeros(len(word_vectors))) for word in sentence])
        else: 
            X_pred = sentence
        
        y = model.predict(X_pred)
        preds =  [inverted_labels_map[pred] for pred in y]
        
        predictions.append(preds)
    """
    for sent in sentences:
        sent = ' '.join(sent)
        y = model.predict(sent)
        predictions.append(y)
    """

    
    return predictions

def evaluation_loop(predicted_labels, true_labels):
    import evaluation
    
    sent_em_list = []
    sent_prec_list = []
    sent_rec_list = []
    sent_f1_list = []

    for pred_sent, true_sent in zip(predicted_labels, true_labels):
        
        predicted_entities = evaluation.extract_entities(pred_sent)
        true_entities = evaluation.extract_entities(true_sent)
        
        # Exact Match Score
        #sent_em = evaluation.exact_match(predicted_entities, true_entities)
        #sent_em_list.append(sent_em)

        # Precision
        sent_prec = evaluation.precision(predicted_entities, true_entities)
        sent_prec_list.append(sent_prec)

        # Recall
        sent_rec = evaluation.recall(predicted_entities, true_entities)
        sent_rec_list.append(sent_rec)

        # F1-Score
        sent_f1 = evaluation.f1_score(predicted_entities, true_entities)
        sent_f1_list.append(sent_f1)

    #avg_em = sum(sent_em_list) / len(sent_em_list)
    avg_prec = sum(sent_prec_list) / len(sent_prec_list)
    avg_rec = sum(sent_rec_list) / len(sent_rec_list)
    avg_f1 = sum(sent_f1_list) / len(sent_f1_list)

    eval_dict = {
            #'avg_em_score': avg_em,
            'precision': avg_prec,
            'recall': avg_rec,
            'f1_score': avg_f1
        }

    return eval_dict

def training_loop(model, data, word_vectors=None):
    import pandas as pd
    from tqdm import tqdm
    
    for sentence, classes in tqdm(zip(data.sentence, data.cls)):
        
        
        if word_vectors is not None:
            X_train = pd.DataFrame([word_vectors[word].values for word in sentence])

        else:
            X_train = sentence
        
        y_train = classes
        
        model.fit(X_train, y_train)