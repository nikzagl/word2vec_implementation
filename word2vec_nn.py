from torch import nn


class W2VNeuralNetwork(nn.Module):
    def __init__(self, embd_len = 300, vocab_len = 1000):
        super().__init__()
        self.embd = nn.Embedding(embedding_dim=embd_len, num_embeddings=vocab_len)
        self.fc1 = nn.Linear(embd_len, vocab_len)

    def forward(self, x):
        out = self.embd(x)
        out = self.fc1(out)
        return out
    
