import pickle
import torch
from models.model import CBOWModel

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    word2id = vocab['word_to_index']
    id2word = vocab['index_to_word']
    return word2id, id2word

def load_model(model_path, vocab_size, embedding_dim=256):
    model = CBOWModel(vocab_size, embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
