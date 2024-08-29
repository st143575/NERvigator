# Script for running the full advanced approach.

# Env. variables
MODEL_NAME='bert-base-uncased'  # models used: 'roberta-base' or 'bert-base-uncased'
EPOCHS=0  # EPOCHS=0 --> zero-shot
DEVICE='cpu'  # torch device
CACHE='cache/model_ckpts/nli'
OUTPUTS='cache/model_outputs/nli'

# Create cache folder structure for model's check points
mkdir -p $CACHE/ner-disease
mkdir -p $CACHE/ner-gene
mkdir -p $CACHE/ner-pol

# Create cache folder structure for model's output
mkdir -p $OUTPUTS/ner-disease
mkdir -p $OUTPUTS/ner-gene
mkdir -p $OUTPUTS/ner-pol

# Create results folder structure
mkdir -p 'results'

# Data preprocessing
python src/scripts/iob2jsonl.py -i datasets -o $CACHE

# Run model
python src/nli/run.py -i $CACHE -o $OUTPUTS -r 'results' -d $DEVICE -m $MODEL_NAME -e $EPOCHS

# Run evaluation
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m $MODEL_NAME -e $EPOCHS