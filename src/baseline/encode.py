import os, math, torch, argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-i', '--input_dir', type=str, default='../../cache/preprocessing_outputs', help="Path to the input data")
    parser.add_argument('-n', '--data_file_name', type=str, choices=['ner-disease', 'ner-gene', 'ner-pol'], required=True, help="Name of the data file")
    parser.add_argument('-o', '--output_dir', type=str, default='../../cache/tfidf_embeddings', help="Path to the output data")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU number")
    return parser.parse_args()

def get_document(data: pd.core.frame.DataFrame):
    """
    Convert the sentence-level data to document-level data.

    Args:
        data: pd.core.frame.DataFrame   A dataframe where each row contains a doc_id, 
                                        a sent_id, the sentence as a list of words,
                                        and the labels of the words in the sentence.

    Returns:
        docid2doc: dict   A dictionary that maps each doc_id to a list of words in the document.
    """
    docid2doc = defaultdict(list)
    for _, row in data.iterrows():
        docid2doc[row['doc_id']] += row['sentence']
    docid2doc = dict(docid2doc)
    return docid2doc

def get_tf(docid2doc: dict, vocab: list):
    """
    Get the frequency of each term from the vocabulary in the input document.
    tf(t, d) = 1 + log10(count(t, d)) if count(t, d) > 0, 0 otherwise
    (Reference: https://web.stanford.edu/~jurafsky/slp3/6.pdf)

    Args:
        docid2doc: dict   A mapping from doc_id to a list of words in the document.
        vocab: list The vocabulary of the collection of documents, i.e. the entire training set.

    Returns:
        term2doctfs: dict  A dictionary that maps each term (i.e. token) to its term frequency in the document.
    """
    term2tfs = dict()
    for term in tqdm(vocab):
        tfs = []
        for doc_id, doc in docid2doc.items():
            tf = doc.count(term)
            if tf > 0:
                tfs.append(1.0 + math.log10(tf))
            else:
                tfs.append(0)
        term2tfs[term] = tfs
    return term2tfs

def get_idf(docid2doc: dict, vocab: list):
    """
    Get the inverse document frequency of each token in the vocabulary.
    idf(t) = log10(N / df(t)), where N is the total number of documents and df(t) is the number of documents in which term t occurs.
    (Reference: https://web.stanford.edu/~jurafsky/slp3/6.pdf)

    Args:
        vocab: list             The vocabulary of the collection of documents, i.e. the entire training set.
        docid2doc: dict         A dictionary that maps each doc_id to a list of tokens in the document.

    Returns:
        term2idf: dict         A dictionary that maps each term (i.e. token) to its inverse document frequency.
    """
    term2idf = dict()
    num_docs = len(docid2doc)
    for term in tqdm(vocab):
        # number of documents containing the term t
        df_t = sum([1 for doc in docid2doc.values() if term in doc])
        term2idf[term] = math.log10(num_docs / df_t)
    return term2idf

def get_tfidf(docid2doc: dict, vocab: list, device: str):
    term2tfs = get_tf(docid2doc, vocab)
    term2idf = get_idf(docid2doc, vocab)
    term2tfidfs = dict()
    term2nonzero_tfidf_indices = dict()
    for term, tf in tqdm(term2tfs.items()):
        tf = torch.tensor(tf, dtype=torch.float32, device=device)
        # print(f"Term: {term}, TF: {tf}, IDF: {term2idf[term]}")
        term2tfidfs[term] = tf * term2idf[term]
        term2tfidfs[term] = term2tfidfs[term].tolist()
        term2nonzero_tfidf_indices[term] = [i for i, val in enumerate(term2tfidfs[term]) if val > 0]
        # print(f"Term: {term}, Non-zero TF-IDF indices: {term2nonzero_tfidf_indices[term]}")
        term2tfidfs[term] = [val for i, val in enumerate(term2tfidfs[term]) if val > 0]
        # print(f"Non-zero TF-IDF scores: {term2tfidfs[term]}\n")
    return term2nonzero_tfidf_indices, term2tfidfs


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the Load preprocessed datasets (format: pd.core.frame.DataFrame).
    ner_dataframe = pd.read_json(f"{input_dir}/{args.data_file_name}/train.jsonl", orient='records', lines=True)

    # Load vocabularies for the training sets.
    vocab = open(f"{input_dir}/{args.data_file_name}/vocab_{args.data_file_name}_train.txt").read().splitlines()
    print(f"Size of the vocabulary vocab_{args.data_file_name}_train: {len(vocab)}")

    # Convert the sentence-level data to document-level data, where
    # each doc_id is mapped to a list of all the words in the document.
    ner_data_docs = get_document(ner_dataframe)
    print(f"Size of {args.data_file_name}_train: {len(ner_data_docs)}")

    # Get the TF-IDF scores for each term in the vocabulary in the input documents.
    # idxs, tfidf_scores = get_tfidf(ner_data_docs, vocab, args.device)
    term2nonzero_tfidf_indices, term2tfidfs = get_tfidf(ner_data_docs, vocab, args.device)
    print(f"Size of the TF-IDF scores: {len(term2tfidfs)}")

    # Add a new column 'tfidf' to the dataframe.
    sent_non_zero_idx_series = []
    sent_tfidf_series = []
    for _, row in tqdm(ner_dataframe.iterrows()):
        non_zero_idx = [term2nonzero_tfidf_indices[word] for word in row['sentence']]
        sent_tfidf = [term2tfidfs[word] for word in row['sentence']]
        sent_non_zero_idx_series.append(non_zero_idx)
        sent_tfidf_series.append(sent_tfidf)

    ner_dataframe['non_zero_idx'] = sent_non_zero_idx_series
    ner_dataframe['tfidf'] = sent_tfidf_series
    ner_dataframe = ner_dataframe[['doc_id', 'sent_id', 'sentence', 'non_zero_idx', 'tfidf', 'tags']]

    # Save the dataframe with the tf-idf scores.
    ner_dataframe.to_json(f"{output_dir}/{args.data_file_name}/train_tfidf.jsonl", orient='records', lines=True)
    print(f"Dataframe with the tf-idf scores has been saved to {output_dir}/{args.data_file_name}/train_tfidf.jsonl")


if __name__ == '__main__':
    main()