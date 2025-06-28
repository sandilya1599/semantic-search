import torch.nn as nn

class CBOWModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
   
    def forward(self,context):
        # input context into embedding layer
        context_vectors = self.embedding(context)
        # take its mean
        context_vectors = context_vectors.mean(dim=1)
        # give the mean into output layer
        logits = self.output(context_vectors)
        return logits


class SkipGramModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
   
    def forward(self,center):
        # input context into embedding layer
        center_vectors = self.embedding(center)
        # give the mean into output layer
        logits = self.output(center_vectors)
        return logits
