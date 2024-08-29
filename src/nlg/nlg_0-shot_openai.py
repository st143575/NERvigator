import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schema import *


os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
langchain_api_key = os.environ['LANGCHAIN_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Natural language generation for NER (0-shot).')
    parser.add_argument('-i', '--input_dir', type=str, default="/mount/studenten-temp1/users/cs/SS2024/cl-team-lab-ner/src/advanced_approach/output")
    parser.add_argument('-dsn', '--dataset_name', type=str, required=True, help="Name of the dataset, should be one of 'ner-disease', 'ner-gene', 'ner-pol'.")
    parser.add_argument('-o', '--output_dir', type=str, default="/mount/studenten-temp1/users/cs/SS2024/cl-team-lab-ner/src/advanced_approach/nlg/output")
    parser.add_argument('-c', '--cache_dir', type=str, default="/mount/studenten-temp1/users/cs/SS2024/cl-team-lab-ner/cache")
    parser.add_argument('-m', '--model', type=str, default="gpt-4o")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    # Load dataset.
    ner_df = pd.read_json(f"{input_dir}/{args.dataset_name}/test.jsonl", orient="records", lines=True)
    ner_sentences = [" ".join(sentence) for sentence in ner_df["sentence"]]

    # Create prompt.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you don't know the value of an attribute asked to extract, "
                "return an empty list for the value of the attribute.",
            ), 
            ("human", "{text}"),
        ]
    )

    # Instantiate LLM.
    llm = ChatOpenAI(
        model=args.model,
        api_key=openai_api_key,
        seed=args.seed,
    )

    # Generate outputs.
    if args.dataset_name == "ner-disease":
        schema = Disease
    elif args.dataset_name == "ner-gene":
        schema = Gene
    elif args.dataset_name == "ner-pol":
        schema = POL
    else:
        raise ValueError(f"Invalid dataset name!")

    runnable = prompt | llm.with_structured_output(schema=schema)

    outputs = []
    for sentence in tqdm(ner_sentences):
        outputs.append(runnable.invoke({"text": sentence}))

    # Add a column for model outputs to the dataframe.
    ner_pred_df = ner_df.copy()
    if args.dataset_name == "ner-disease" or args.dataset_name == "ner-gene":
        ner_pred_df['model_output'] = [ne.names for ne in outputs]
    elif args.dataset_name == "ner-pol":
        ner_pred_df['model_output'] = [{"PERSONS": ne.persons, "ORGANIZATIONS": ne.organizations, "LOCATIONS": ne.locations, "MISCS": ne.miscs} for ne in outputs]
    else:
        raise ValueError(f"Invalid dataset name!")

    # Save the model outputs.
    ner_pred_df.to_json(f"{output_dir}/{args.dataset_name}-{args.model}-0-shot.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()