import argparse
import os
import time

import jieba

jieba.disable_parallel()

from typing import List
from tokenizers import Tokenizer, NormalizedString, PreTokenizedString
from tokenizers.models import ChineseWordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer

from datasets import load_dataset, set_caching_enabled
set_caching_enabled(False)

from functools import partial

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="./example_data", type=str, help='Path to input TXT')

    group = parser.add_argument_group(title='vocab path')
    group.add_argument('--vocab_path', default="./", type=str, help='Path of vocab_file')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_file", default="./processed_data.npz", type=str)

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=16,
                       help='Number of worker processes to launch')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0

    return args


def getfiles(path, except_pattern='py'):
    file_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if len(except_pattern) > 0 and except_pattern in name:
                continue
            file_list.append(os.path.join(root, name))
    return file_list


class JiebaPreTokenizer:
    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        return [normalized_string[w[1]: w[2]] for w in jieba.tokenize(str(normalized_string))]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.jieba_split)


def tokenize(_tokenizer, x):
    res = _tokenizer.encode_batch(x['text'])
    return {
        'id': [r.ids for r in res],
        'attention_mask': [r.attention_mask for r in res]
    }


if __name__ == '__main__':
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin_list = getfiles(args.input, except_pattern='git')
    print("fin_list: ", fin_list)

    tokenizer = Tokenizer(ChineseWordPiece(vocab=os.path.join(args.vocab_path, 'vocab.txt'), unk_token='<unk>'))
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.enable_padding(pad_id=53225, pad_token="[PAD]", length=2048)
    tokenizer.enable_truncation(max_length=2048)
    jieba_pre_tokenizer = JiebaPreTokenizer()
    #    tokenizer.pre_tokenizer = PreTokenizer.custom(jieba_pre_tokenizer)

    dataset = load_dataset('text', data_files={'train': fin_list}, keep_in_memory=True)

    map_func = partial(tokenize, tokenizer)
    print(dataset)
    print(dataset['train'])
    tokenized_data = dataset['train'].map(
        map_func,
        num_proc=args.workers,
        batched=True,
        batch_size=64,
        keep_in_memory=True,
        load_from_cache_file=False
    )
    print('[Stage 1] Processed in %.2fs' % (time.time() - startup_start))
    # tokenized_data.save_to_disk('./preprocessed_data/')  # File too large
    ids = tokenized_data['id']
    attention_masks = tokenized_data['attention_mask']
    np.savez_compressed(args.output_file,
                        id=np.asarray(ids, dtype=np.uint16),
                        attention_mask=np.asarray(attention_masks, dtype=np.uint16)
                        )
    print(np.asarray(ids).shape)
    print('Saved processed data to ', args.output_file)

    print("Total time to used:", time.time() - startup_start)
