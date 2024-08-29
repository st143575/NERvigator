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