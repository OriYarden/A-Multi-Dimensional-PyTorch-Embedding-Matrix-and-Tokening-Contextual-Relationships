import torch
from torch import nn


class EmbeddingMatrix(nn.Module):
    """ PyTorch Embedding Matrix.

    Text embeddings of a given word (token) along with its
    inter-relationships in context to other words (tokens).
    """
    def __init__(self, vocab_size=1):
        super().__init__()
        for n in range(vocab_size):
            exec(
                f'self.embed_layer_{n} = nn.Embedding(num_embeddings={vocab_size}, embedding_dim={vocab_size})'
            )
        self.vocab_size = vocab_size
        self.vocab_indexes = [i for i in range(vocab_size)]

    def forward(self, x):
        word_index = x[0]
        contextual_indexes = self.vocab_indexes if x.shape[0] == 1 else x[1:]
        outputs = []
        for i in contextual_indexes:
            y = eval(
                f'self.embed_layer_{word_index}(torch.LongTensor([{i}]))'
            )
            outputs.append(y)
        return torch.stack(outputs, dim=0).squeeze()


if __name__ == '__main__':
    vocab_size = 256
    model = EmbeddingMatrix(
        vocab_size=vocab_size,
    )
    x = torch.tensor([0, 2, 3])
    y = model(x)
    print(y.shape)

