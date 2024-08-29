# Script for running the full baseline approach.

# Env. variables
DEVICE='cpu' # torch device
CACHE='cache/model_ckpts/baseline'
OUTPUTS='cache/model_outputs/baseline'

# Create cache folder structure for model's check points
mkdir -p $CACHE/ner-disease
mkdir -p $CACHE/ner-gene
mkdir -p $CACHE/ner-pol

# Create cache folder structure for model's output
mkdir -p $OUTPUTS/ner-disease
mkdir -p $OUTPUTS/ner-gene
mkdir -p $OUTPUTS/ner-pol

# Create results folder structure
mkdir -p results

# Data preprocessing
python src/scripts/iob2jsonl.py -i datasets -o cache/model_ckpts/baseline

# Encode the documents with tf-idf scores.
python src/baseline/encode.py -n ner-disease -i $CACHE -o $CACHE --device $DEVICE
python src/baseline/encode.py -n ner-gene -i $CACHE -o $CACHE --device $DEVICE
python src/baseline/encode.py -n ner-pol -i $CACHE -o $CACHE --device $DEVICE

# Run the baseline
python src/baseline/run.py -i $CACHE -o $OUTPUTS -r results

