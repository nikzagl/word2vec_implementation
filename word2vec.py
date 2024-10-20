import torch
from torch import nn
from torch import optim
import word2vec_nn
import re
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DataWrapper(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])


def __prepare_data(text, window_size = 5):
    text = re.sub(r'[^a-z@# ]', '', text.lower())
    tokens = text.lower().split()
    idx_2_word = {i: word for i, word in enumerate(set(tokens))}
    word_2_idx = {word: i for i, word in enumerate(set(tokens))}
    context_tuples = list()
    for i, word in enumerate(tokens):
        left_border = max(0, i-window_size)
        right_border = min(i+window_size, len(tokens))
        for j in range(left_border, right_border):
            if i != j :
                context_tuples.append((word_2_idx[word], word_2_idx[tokens[j]]))
    return tokens, word_2_idx, torch.tensor(context_tuples, dtype = torch.long)


def get_embeddings(text, embd_len = 50, window_size = 5, batch_size = 1, num_epochs = 5):
    tokens, word_2_idx, data = __prepare_data(text, window_size=window_size)
    X = data[:, 0]
    del data
    dataset = DataWrapper(X, y)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    model = word2vec_nn.W2VNeuralNetwork(embd_len, len(word_2_idx))
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
    matrix = next(model.embd.parameters()).detach().numpy()
    words = {word: matrix[word_2_idx[word]] for word in tokens}
    return words

f = open("1.txt")
s = " ".join(f.readlines()[:100])
words = get_embeddings(s)

cosine_similarity = lambda x, y: np.dot(x, y)/np.linalg.norm(x)/np.linalg.norm(y)
def most_similar(word, words):
    w = sorted(list(words), key = lambda word_to: cosine_similarity(words[word], words[word_to]), reverse=True)
    return w
print(len(words))
print(most_similar("a", words)[:10])