import jieba
from collections import Counter

def build_vocab_from_file(filepath):
    with open("sentences.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 分词
    tokenized_sentences = [list(jieba.cut(line.strip())) for line in lines]

    # 构建词表（词频统计）
    word_counts = Counter()
    for sent in tokenized_sentences:
        word_counts.update(sent)

    # 按词频排序
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # 构建词到索引映射
    word_to_idx = {word: idx for idx, (word, _) in enumerate(sorted_words)}

    # 从索引到词的映射
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word,tokenized_sentences

