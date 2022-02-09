import numpy as np
from datasets import load_dataset
from tokenizers.implementations import BertWordPieceTokenizer
from tqdm import tqdm

if __name__ == '__main__':
    dataset = load_dataset("web_of_science", 'WOS46985')
    text = dataset['train']['input_data']
    tokenizer = BertWordPieceTokenizer(
        vocab='./vocab.txt',
        unk_token='<unk>',
    )
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=2048)
    tokenizer.enable_truncation(max_length=2048)
    ids = []
    for line in tqdm(text):
        ids.append(tokenizer.encode(line).ids)
    ids = np.stack(ids)
    print(ids.shape)
    np.savez_compressed('./processed_data.npz', ids=ids)
