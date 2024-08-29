# Script for running the full baseline approach.

# Env. variables
DEVICE='cpu' # torch device
CACHE='cache/model_ckpts/baseline-v2'
OUTPUTS='cache/model_outputs/baseline-v2'

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
python src/scripts/iob2csv.py -i datasets -o $CACHE

# Run the baseline
python src/baseline-v2/run.py -i $CACHE -o $OUTPUTS -r results