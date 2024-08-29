import os, argparse
import pandas as pd
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-i', '--input_dir', type=str, default='../../datasets', help="Path to the input data")
    parser.add_argument('-o', '--output_dir', type=str, default='../../cache/model_ckpt', help="Path to the output data")
    return parser.parse_args()


def load_dataset(data_path_name: str):
    """
    Load a dataset from an .iob file and store the data in a list.

    Args:
        data_path_name: str     The path to the .iob file and the file name, e.g. 
                                /cl-team-lab-ner/datasets/ner-disease/train.iob

    Returns:
        docid2word_tags: dict(list(tuple))      A dictionary with the document ID (doc_id) as key and 
                                                a list of tuples with the word and tag as value.
    """
    docid2word_tags = dict()
    with open(data_path_name, "r") as file:
        docid = 0
        for line in file:
            line = line.strip()
            if line == "":
                docid += 1
                continue
            if line.startswith('|'):
                line = "\t" + line
            word, tag = line.split("\t|")
            if docid not in docid2word_tags:
                docid2word_tags[docid] = []
            docid2word_tags[docid].append((word, tag))

    return docid2word_tags


def segment(docid2word_tags: dict):
    """
    Segment the input document into sentences using the following heuristic:
    A sentence starts with an uppercase letter and ends with a period.

    Args:
        docid2word_tags: A dictionary that maps a document ID (doc_id) to a list of (word, tag) tuples.

    Returns:
        sentence_dict: A dictionary that maps a document ID (doc_id) to a list of sentences.
    """
    segmented_documents = []
    for _, words_tags in docid2word_tags.items():
        doc_sentences = []
        doc_sentence = []
        for i, (word, tag) in enumerate(words_tags):
            doc_sentence.append((word, tag))

            # If the current token is a period and the next word starts with an uppercase letter, 
            # then the current token is the end of a sentence.
            if word == '.' and (i + 1 < len(words_tags) and words_tags[i + 1][0][0].isupper()):
                doc_sentences.append(doc_sentence)
                doc_sentence = []

        # Append any remaining words as the last sentence if not empty.
        if doc_sentence:
            doc_sentences.append(doc_sentence)
        
        sentences_tags = {
            'sentences': [[word for word, tag in sentence] for sentence in doc_sentences],
            'tags': [[tag for word, tag in sentence] for sentence in doc_sentences]
        }
        segmented_documents.append(sentences_tags)
        
    return segmented_documents


def convert_to_dataframe(segmented_documents):
    """
    Convert the segmented documents to a pandas DataFrame.

    Args:
        segmented_documents: list(dict)     A list of dictionaries containing a list of sentences and 
                                            a list of corresponding tags.

    Returns:
        segmented_documents_df: pandas.DataFrame    A pandas DataFrame containing the sentences and tags.
    """
    data = []
    for doc_id, sentid2word_tags in enumerate(segmented_documents):
        sentences = sentid2word_tags['sentences']
        tags = sentid2word_tags['tags']
        assert len(sentences) == len(tags)
        for sent_id, sentence in enumerate(sentences):
            data.append([doc_id, sent_id, sentence, tags[sent_id]])

    segmented_documents_df = pd.DataFrame(data, columns=["doc_id", "sent_id", "sentence", "tags"])
    return segmented_documents_df


def get_vocabulary(dataset):
    vocabulary = set()
    for value in dataset.values():
        for word, _ in value:
            vocabulary.add(word)
    return vocabulary


def create_labels_map(labeled_sentences: list):
        cls = 0
        labels_map = {'O': cls}
        for labeled_sent in labeled_sentences:
            for label in labeled_sent:
                if label not in labels_map:
                    cls += 1
                    labels_map[label] = cls
        
        return labels_map

def map_sentence_labels(labels_map, sentence_labels: list):
        return [labels_map[label] for label in sentence_labels]


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")

    for subdir, dirs, files in os.walk(input_dir):
        dataset_name = Path(subdir).name
        # Innitialize label mapping on corpus level
        cls = 0 
        labels_map = {'O': cls}

        def map_sent(sentence_labels): 
                    return [labels_map[label] for label in sentence_labels]
        
        for file in files:
            if file.endswith('.iob'):
                #dataset_name = Path(subdir).name
                data_split = file.split('.')[0]
                print(f"Start preprocessing dataset {dataset_name}_{data_split}...")
                docid2word_tags = load_dataset(f"{subdir}/{file}")
                segmented_documents = segment(docid2word_tags)
                segmented_documents_df = convert_to_dataframe(segmented_documents)
                
                # Label mapping
                for labeled_sent in segmented_documents_df.tags:
                    for label in labeled_sent:
                        if label not in labels_map:
                            cls += 1
                            labels_map[label] = cls
                
                segmented_documents_df['cls'] = segmented_documents_df.tags.map(map_sent)
                
                # Save in cache
                segmented_documents_df.to_json(f"{output_dir}/{dataset_name}/{data_split}.jsonl", orient='records', lines=True)
                print(f"Preprocessing of dataset {dataset_name}_{data_split} done. Output saved to {output_dir}/{dataset_name}/{data_split}.json\n")

                # Get vocabularies from the train/dev/dev-predicted/test splits.
                vocabulary = get_vocabulary(docid2word_tags)
                with open(f"{output_dir}/{dataset_name}/vocab_{dataset_name}_{data_split}.txt", "w") as file:
                    for word in vocabulary:
                        file.write(f"{word}\n")
                print(f"Vocabulary for {dataset_name}_{data_split} saved to {output_dir}/{dataset_name}/vocabulary.txt\n")
            
            labels_map_t = {value: key for key, value in labels_map.items()}
            labels_map_df = pd.DataFrame.from_dict([labels_map_t])
            #labels_map_df.to_json(f"{output_dir}/{dataset_name}/labels_map.jsonl", orient='records', lines=True)

            


if __name__ == "__main__":
    main()