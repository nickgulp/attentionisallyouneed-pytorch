import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocal_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocal_size)

    def forward(self, x):
        return self.embedding