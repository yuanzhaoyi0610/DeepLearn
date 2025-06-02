import torch.nn as nn

#构建一个简单的CBOW模型
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_words):
        embeds = self.embedding(context_words)         # (batch_size, context_size, embed_dim)
        avg_embeds = embeds.mean(dim=1)                 # (batch_size, embed_dim)
        out = self.linear(avg_embeds)                   # (batch_size, vocab_size)
        return out