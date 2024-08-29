import random

class RandomNER:

    def __init__(self, labels, seed=69):
        self.labels = labels
        self.seed = seed
    
    @staticmethod
    def extract_entities(sentence_labels):
        entities = []
        current_entity = None
        start_index = None

        for i, label in enumerate(sentence_labels):
            if label.startswith('B-'):
                # New entity starts
                if current_entity:
                    current_entity['end'] = i - 1
                    entities.append(current_entity)
                current_entity = {'type': label[2:], 'start': i, 'end': None}
            elif label.startswith('I-'):
                # Continuing the current entity
                if current_entity:
                    if not start_index:
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
            current_entity['end'] = len(sentence_labels) - 1
            entities.append(current_entity)

        return entities
    
    @staticmethod
    def serialize_labels(labeled_sentences: list):

        labels = set([label for labeled_sentence in labeled_sentences for label in labeled_sentence])

        return {key: label for key, label in enumerate(labels)}


    def predict(self, X):
        random.seed(self.seed)
        
        #tokens = sentence.split()
        
        # makes random predictions of IOB labels
        y_labels = [random.choice(self.labels) for sample in X]
        
        return y_labels




if __name__ == '__main__':
    import evaluation

    labels = ['I-DISEASE', 'B-DISEASE', 'O']
    sentence = 'John was diagnosed with influenza last winter, but he recovered quickly. Unfortunately, he developed pneumonia shortly afterward and had to be hospitalized.'
    y = ['O', 'O', 'O', 'O', 'B-DISEASE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-DISEASE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    
    y_pred = RandomNER(labels).predict(sentence)
    y_pred_entities = RandomNER.extract_entities(y_pred)
    y_entities = RandomNER.extract_entities(y)
    
    exact_match = evaluation.exact_match(y_pred_entities, y_pred_entities)
    print('EM:', exact_match)

    exact_match = evaluation.exact_match(y_entities, y_entities)
    print('EM:', exact_match)

    exact_match = evaluation.exact_match(y_pred_entities, y_entities)
    print('EM:', exact_match)

    
