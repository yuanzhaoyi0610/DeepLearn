import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import CBOW
from vocab import build_vocab_from_file  # 导入词表构建函数

word_to_idx, idx_to_word,tokenized_sentences = build_vocab_from_file("sentences.txt")


# 参数
V = len(word_to_idx)
N = 64  # 词向量维度
window_size = 2  # 窗口大小，以当前词为中心的左右两侧的词
epoch = 100

# 构造 CBOW 训练样本  格式为:([46,47,6,0],5) 上2个字和下两个字的index，以及标签值的index
data = []
for sentence in tokenized_sentences:
    indexed = [word_to_idx[word] for word in sentence if word in word_to_idx]
    for i in range(window_size, len(indexed) - window_size):
        context = indexed[i - window_size:i] + indexed[i+1:i + window_size + 1]
        target = indexed[i]
        data.append((context, target))

print("训练样本数量:", len(data))
print("data:",data)

# 把训练数据转成tensor
contexts = torch.tensor([x[0] for x in data])
targets = torch.tensor([x[1] for x in data])

# 创建 Dataset 和 DataLoader
dataset = TensorDataset(contexts, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # 你可以调整 batch_size

# 创建模型和优化器
model = CBOW(V, N)
optimizer = optim.SGD(model.parameters(), lr=0.1)


# 训练100轮
for e in range(epoch):
    model.train()
    total_loss = 0

    for batch_contexts, batch_targets in dataloader:
        optimizer.zero_grad()
        logits = model(batch_contexts)        # (batch_size, vocab_size)
        loss = F.cross_entropy(logits, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if e % (epoch//10) == 0:
        print(f"Epoch {e}, Loss: {total_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "cbow.pth")