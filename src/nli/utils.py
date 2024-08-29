import csv, json

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

def save_entities_to_csv(entities, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Type', 'Start', 'End'])  # Writing the header
        for entity in entities:
            writer.writerow([entity['type'], entity['start'], entity['end']])


def save_entities_to_json(predicted_entities, true_entities, filename):
    data = {
        'predicted_entities': predicted_entities,
        'true_entities': true_entities
    }
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def read_entities_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        predicted_entities = data['predicted_entities']
        true_entities = data['true_entities']
    return predicted_entities, true_entities


def read_entities_from_csv(filename):
    entities = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            entity = {'type': row[0], 'start': int(row[1]), 'end': int(row[2])}
            entities.append(entity)
    return entities