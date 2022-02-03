import collections

import jieba
import numpy as np
from tqdm import tqdm
from tokenizers.implementations import BertWordPieceTokenizer


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf-8') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def do_tokenize():
    with open('../example_data/001.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tokenizer = BertWordPieceTokenizer(vocab='./vocab.txt')
    for line in tqdm(lines):
        line = line.strip()
        if line != '':
            _res = np.zeros(shape=(3072, ), dtype=np.int32)
            i = 0
            for x in jieba.cut(line, cut_all=False):
                # _x = tokenizer.token_to_id(x)
                # if _x is not None:
                #     _res[i] = _x
                #     i += 1
                # if i == 3072:
                #     break
                try:
                    _res[i] = res[x]
                    i += 1
                    if i == 3072:
                        break
                except KeyError:
                    pass


if __name__ == '__main__':
    res = load_vocab('./vocab.txt')
    print(len(res))
    do_tokenize()
