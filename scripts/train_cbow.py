import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from utils.train_pipeline import get_tokenized_corpus
from models.model import CBOWModel, SkipGramModel
from utils.vocab_utils import build_vocab
from utils.dataset_utils import generate_training_set
from utils.map_utils import create_mappings, build_id_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import os
from config import CORPUS_PATH, EMBEDDING_DIM, WINDOW_SIZE, BATCH_SIZE, EPOCHS

# Download required data once
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Directory to save models and vocabs
SAVE_DIR = 'data'
os.makedirs(SAVE_DIR, exist_ok=True)

# Utility: Get similar words from embedding
def get_similar_words(word, model, word2id, id2word, top_k=5):
    embeddings = model.embedding.weight.data
    word_id = word2id.get(word, word2id.get('<UNK>', 0))
    target_vector = F.normalize(embeddings[word_id].unsqueeze(0), dim=1)
    norm_embeddings = F.normalize(embeddings, dim=1)
    cosine_sim = torch.matmul(norm_embeddings, target_vector.T).squeeze()
    top_ids = torch.topk(cosine_sim, top_k + 1).indices
    return [id2word[i.item()] for i in top_ids if i.item() != word_id][:top_k]

# Model training function
def train_model(epochs, context_tensors, target_tensors, batch_size, model, device, lr=0.01):
    train_data_set = TensorDataset(context_tensors, target_tensors)
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for context_batch, target_batch in train_loader:
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(context_batch)
            loss = criteria(logits, target_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    return model

# Save model and vocab to data directory
def save_model_and_mapping(model, word2id, id2word):
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cbow_model_lem.pt"))
    with open(os.path.join(SAVE_DIR, "cbow_vocab_lem.pkl"), "wb") as f:
        pickle.dump({
            "word_to_index": word2id,
            "index_to_word": id2word
        }, f)

def prepare_data(corpus_path, window_size):
    tokenized_corpus = get_tokenized_corpus(corpus_path)
    vocab = build_vocab(tokenized_corpus)
    word2id, id2word = create_mappings(vocab=vocab)
    id_seq = build_id_sequences(tokenized_corpus, word2id)
    context_ids, target_ids = generate_training_set(id_seq, window_size)
    return context_ids, target_ids, word2id, id2word

def train_and_save(context_ids, target_ids, word2id, id2word, embedding_dim, device, batch_size, epochs):
    vocab_size = max(word2id.values()) + 1
    model = CBOWModel(vocab_size, embedding_dim).to(device)
    combined = list(zip(context_ids, target_ids))
    random.shuffle(combined)
    context_ids, target_ids = zip(*combined)
    context_ids = list(context_ids)
    target_ids = list(target_ids)
    context_tensors = torch.tensor(context_ids, dtype=torch.long).to(device)
    target_tensors = torch.tensor(target_ids, dtype=torch.long).to(device)
    model = train_model(model=model, epochs=epochs, batch_size=batch_size,
                        context_tensors=context_tensors, target_tensors=target_tensors, device=device)
    save_model_and_mapping(model, word2id, id2word)
    return model

def interactive_similarity(model, word2id, id2word):
    text = input("Enter a word to find similar words: ")
    print(get_similar_words(text.lower(), model, word2id, id2word))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    context_ids, target_ids, word2id, id2word = prepare_data(CORPUS_PATH, WINDOW_SIZE)
    model = train_and_save(context_ids, target_ids, word2id, id2word, EMBEDDING_DIM, device, BATCH_SIZE, EPOCHS)
    interactive_similarity(model, word2id, id2word)

if __name__ == "__main__":
    main()