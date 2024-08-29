DATASETS = ['ner-disease', 'ner-gene', 'ner-pol']
#DATASETS = ['ner-pol']
GPU = 1



label2id = {
    'ner-disease': {
        'O': 0, 
        'B-DISEASE': 1,
        'I-DISEASE': 2,
        'PAD': 3
    },
    'ner-gene': {
        'O': 0, 
        'B-PROTEIN': 1,
        'I-PROTEIN': 2,
        'PAD': 9
    },
    'ner-pol':{
        'O': 0, 
        'B-PER': 1, 
        'I-PER': 2, 
        'B-ORG': 3,
        'I-ORG': 4,
        'B-LOC': 5,
        'I-LOC': 6,
        'B-MISC': 7,
        'I-MISC': 8,
        'PAD': 9
    }
}

id2label = {
    'ner-disease': {
        0: 'O', 
        1: 'B-DISEASE',
        2: 'I-DISEASE',
        3: 'PAD'
    },
    'ner-gene': {
        0: 'O', 
        1: 'B-PROTEIN',
        2: 'I-PROTEIN',
        3: 'PAD'
    },
    'ner-pol': {
        0: 'O', 
        1: 'B-PER', 
        2: 'I-PER', 
        3: 'B-ORG',
        4: 'I-ORG',
        5: 'B-LOC',
        6: 'I-LOC',
        7: 'B-MISC',
        8: 'I-MISC',
        9: 'PAD'
    }
}

PROMPTS_TEMPLATES = {
    'void' : "{sentence}{token}".strip(),
    'basic': "Label '{token}' in '{sentence}'".strip(),
    'standard': "Identify the entity type for '{token}' in the context of '{sentence}'.".strip()
}