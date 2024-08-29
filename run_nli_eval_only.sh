# Env vars
CACHE='cache/model_ckpts/nli'
OUTPUTS='cache/model_outputs/nli'

# Run evaluation
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'bert-base-uncased' -e 0
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'bert-base-uncased' -e 1
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'bert-base-uncased' -e 3
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'bert-base-uncased' -e 5
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'bert-base-uncased' -e 7
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'bert-base-uncased' -e 10

python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'roberta-base' -e 0
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'roberta-base' -e 1
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'roberta-base' -e 3
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'roberta-base' -e 5
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'roberta-base' -e 7
python src/nli/evaluation.py -i $OUTPUTS -o 'results' -m 'roberta-base' -e 10