# Env vars
INPUT_DIR='cache/nlg/input'
NLG_OUTPUT_DIR='cache/nlg/output'
EVAL_OUTPUT_DIR='cache/nlg/evaluation/output'

# Run evaluation
#python src/nlg/evaluation.py -i $OUTPUTS -o 'results' -m 'bert-base-uncased' -e 0
python src/nlg/evaluate_nlg.py -i $NLG_OUTPUT_DIR -o $EVAL_OUTPUT_DIR -dsn ner-disease -m gpt-4o
python src/nlg/evaluate_nlg.py -i $NLG_OUTPUT_DIR -o $EVAL_OUTPUT_DIR -dsn ner-gene -m gpt-4o
python src/nlg/evaluate_nlg.py -i $NLG_OUTPUT_DIR -o $EVAL_OUTPUT_DIR -dsn ner-pol -m gpt-4o
