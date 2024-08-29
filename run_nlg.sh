# Script for running the full advanced approach.

# Env. variables
#MODEL_NAME='bert-base-uncased'  # models used: 'roberta-base' or 'bert-base-uncased'
#EPOCHS=0  # EPOCHS=0 --> zero-shot
#DEVICE='cpu'  # torch device
INPUT_DIR='cache/nlg/input'
NLG_OUTPUT_DIR='cache/nlg/output'
EVAL_OUTPUT_DIR='cache/nlg/evaluation/output'

# Create cache folder structure for model's check points
#mkdir -p $CACHE/ner-disease
#mkdir -p $CACHE/ner-gene
#mkdir -p $CACHE/ner-pol

# Create cache folder structure for model's output
#mkdir -p $OUTPUTS/ner-disease
#mkdir -p $OUTPUTS/ner-gene
#mkdir -p $OUTPUTS/ner-pol

# Create results folder structure
#mkdir -p 'results'

# Data preprocessing
#python src/scripts/iob2jsonl.py -i datasets -o $CACHE

# Run model
#python src/nlg/run.py -i $CACHE -o $OUTPUTS -r 'results' -d $DEVICE -m $MODEL_NAME -e $EPOCHS
python src/nlg/nlg_0-shot_openai.py -i $INPUT_DIR -o $NLG_OUTPUT_DIR -dsn ner-disease -m gpt-4o
python src/nlg/nlg_0-shot_openai.py -i $INPUT_DIR -o $NLG_OUTPUT_DIR -dsn ner-gene -m gpt-4o
python src/nlg/nlg_0-shot_openai.py -i $INPUT_DIR -o $NLG_OUTPUT_DIR -dsn ner-pol -m gpt-4o

# Run postprocessing
python postprocessing.py -i $NLG_OUTPUT_DIR -o $NLG_OUTPUT_DIR -dsn ner-disease
python postprocessing.py -i $NLG_OUTPUT_DIR -o $NLG_OUTPUT_DIR -dsn ner-gene
python postprocessing.py -i $NLG_OUTPUT_DIR -o $NLG_OUTPUT_DIR -dsn ner-pol

# Run evaluation
#python src/nlg/evaluation.py -i $OUTPUTS -o 'results' -m $MODEL_NAME -e $EPOCH
python src/nlg/evaluate_nlg.py -i $NLG_OUTPUT_DIR -o $EVAL_OUTPUT_DIR -dsn ner-disease -m gpt-4o
python src/nlg/evaluate_nlg.py -i $NLG_OUTPUT_DIR -o $EVAL_OUTPUT_DIR -dsn ner-gene -m gpt-4o
python src/nlg/evaluate_nlg.py -i $NLG_OUTPUT_DIR -o $EVAL_OUTPUT_DIR -dsn ner-pol -m gpt-4o
