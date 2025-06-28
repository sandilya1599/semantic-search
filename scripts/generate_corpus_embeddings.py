import pickle
import torch
from models.model import CBOWModel
from utils.train_pipeline import get_tokenized_corpus
from semantic_search.search import embed_corpus, save_corpus_embeddings
from utils.model_utils import load_vocab, load_model
from config import VOCAB_PATH, MODEL_PATH, CORPUS_EMB_PATH, EMBEDDING_DIM, CORPUS_PATH

def generate_embeddings(corpus_path, vocab_path, model_path, output_path, embedding_dim=256):
    tokenised_corpus = get_tokenized_corpus(corpus_path)
    word2id, _ = load_vocab(vocab_path)
    vocab_size = max(word2id.values()) + 1
    model = load_model(model_path, vocab_size, embedding_dim)
    corpus_embeddings = embed_corpus(tokenised_corpus, model, word2id)
    save_corpus_embeddings(corpus_embeddings, output_path)
    print(f"Corpus embeddings saved to {output_path}")

def main():
    generate_embeddings(CORPUS_PATH, VOCAB_PATH, MODEL_PATH, CORPUS_EMB_PATH, EMBEDDING_DIM)

if __name__ == "__main__":
    main()
