from .document_reader import read_documents_stream
from .text_preprocessor import pre_process_text_stream, tokenize_stream
from config import CORPUS_PATH

def get_tokenized_corpus(path: str = None):
    if path is None:
        path = CORPUS_PATH
    document = read_documents_stream(path)
    sentence_stream = pre_process_text_stream(document)
    tokenized_corpus = list(tokenize_stream(sentence_stream))
    return tokenized_corpus

if __name__ == "__main__":
    print(get_tokenized_corpus())