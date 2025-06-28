from utils.text_preprocessor import tokenize
from semantic_search.search import embed_sentence, load_corpus_embeddings, semantic_search
from utils.model_utils import load_vocab, load_model
from config import VOCAB_PATH, MODEL_PATH, CORPUS_EMB_PATH

def process_query(text, model, word2id, corpus_embeddings, top_k=11):
    tokenized_query = tokenize(text)
    print(tokenized_query)
    query_embedding = embed_sentence(tokenized_query, model, word2id)
    results = semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    return results


def main():
    word2id, _ = load_vocab(VOCAB_PATH)
    vocab_size = max(word2id.values()) + 1
    model = load_model(MODEL_PATH, vocab_size)
    corpus_embeddings = load_corpus_embeddings(CORPUS_EMB_PATH)
    text = input('Enter your query: ')
    results = process_query(text, model, word2id, corpus_embeddings, top_k=11)
    print(results)


if __name__ == "__main__":
    main()
