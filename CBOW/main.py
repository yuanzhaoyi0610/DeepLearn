import torch
from model import CBOW
from vocab import build_vocab_from_file
from torch.nn import functional as F
import jieba

#关闭jieba日志，避免输出不清晰
jieba.setLogLevel(jieba.logging.WARN)

# 加载词表
word_to_idx, idx_to_word,_ = build_vocab_from_file("sentences.txt")
vocab_size = len(word_to_idx)
embed_dim = 64  # 与训练时保持一致

# 加载模型
model = CBOW(vocab_size, embed_dim)
model.load_state_dict(torch.load("cbow.pth"))
model.eval()

# 预测函数，输入batch上下文词
def predict_center_words(model, context_words):
    model.eval()
    with torch.no_grad():
        logits = model(context_words)
        probs = F.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1)
    return predicted_idx, probs


#输入句子，得到相对应的分词索引列表
def get_context_index(sentences, word_to_idx, window_size=2):
    tokenized_sentences = [list(jieba.cut(sentence.strip())) for sentence in sentences]
    indexed = []
    for sentence in tokenized_sentences:
        indexed.append([word_to_idx[word] for word in sentence if word in word_to_idx])
    return indexed


#测试样本
test_context_words=["我爱处理","机器学习人工智能","学习是的"]

#得到测试样本句子分词，并转换成索引
test_context_idx=get_context_index(test_context_words, word_to_idx)

#索引list转换为tensor
test_context_words_tensor = torch.tensor(test_context_idx)

#预测
predicted, _ = predict_center_words(model, test_context_words_tensor)

#打印预测结果
for i, idx in enumerate(predicted):
    Word = ''
    for j in range(3):
        if j==2:
            Word+=idx_to_word[idx.item()]
        Word+=idx_to_word[test_context_words_tensor[i,j].item()]
    print(f"第{i+1}个样本:测试句子:{test_context_words[i]};预测中心词索引为:{idx.item()};预测中心词为:({idx_to_word[idx.item()]});加上中心词后句子:{Word}")

