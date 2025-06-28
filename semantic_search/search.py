import torch
import torch.nn.functional as F
import pickle

# Embeds a single sentence and returns the normalized mean embedding
def embed_sentence(sentence, model, word2id):
    current_sentence = [word2id.get(token, 0) for token in sentence]
    current_sentence = torch.tensor(current_sentence, dtype=torch.long)
    with torch.no_grad():
        sentence_embedding = model.embedding(current_sentence)
    aggregated_sum = sentence_embedding.mean(dim=0)
    aggregated_sum = F.normalize(aggregated_sum.unsqueeze(0), dim=1).squeeze(0)
    return aggregated_sum

# Embeds a corpus (list of tokenized sentences)
def embed_corpus(tokenized_corpus, model, word2id):
    corpus_embeddings = []
    for sentence in tokenized_corpus:
        emb = embed_sentence(sentence, model, word2id)
        corpus_embeddings.append((emb, sentence))
    return corpus_embeddings

# Save corpus embeddings to a pickle file
def save_corpus_embeddings(corpus_embeddings, path):
    with open(path, "wb") as f:
        pickle.dump({"corpus_embeddings": corpus_embeddings}, f)

# Load corpus embeddings from a pickle file
def load_corpus_embeddings(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["corpus_embeddings"]

# Perform semantic search: returns top_k most similar sentences to the query embedding
def semantic_search(query_embedding, corpus_embeddings, top_k=10):
    similarities = []
    for embedding, sentence in corpus_embeddings:
        sim = torch.matmul(embedding, query_embedding).item()
        similarities.append((sim, sentence))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]
