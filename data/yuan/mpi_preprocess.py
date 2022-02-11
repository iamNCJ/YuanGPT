from typing import List

import jieba
from mpi4py import MPI
import numpy
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer

from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer


class JiebaPreTokenizer:
    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        return [normalized_string[w[1]: w[2]] for w in jieba.tokenize(str(normalized_string))]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.jieba_split)


tokenizer = Tokenizer(WordPiece(vocab='vocab.txt', unk_token='<unk>'))
tokenizer.pre_tokenizer = BertPreTokenizer()
tokenizer.enable_padding(pad_id=53225, pad_token="[PAD]", length=2048)
tokenizer.enable_truncation(max_length=2048)
jieba_pre_tokenizer = JiebaPreTokenizer()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size

if rank == 0:
    lines = []
    with open('example_data/001.txt', 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() != '':
                lines.append(line)
    avg_len = len(lines) // size
    lines = [lines[r * avg_len: (r + 1) * avg_len] for r in range(size)]
else:
    lines = None

lines = comm.scatter(lines, root=0)
print(rank, lines)
