import numpy as np
from datasets import load_dataset
from tokenizers.implementations import BertWordPieceTokenizer
from functools import partial


def tokenize(_tokenizer, x):
    res = _tokenizer.encode_batch(x['input_data'])
    return {
        'id': [r.ids for r in res],
        'attention_mask': [r.attention_mask for r in res]
    }


if __name__ == '__main__':
    tokenizer = BertWordPieceTokenizer(
        vocab='./vocab.txt',
        unk_token='<unk>',
    )
    tokenizer.enable_padding(pad_id=53225, pad_token="[PAD]", length=2048)
    tokenizer.enable_truncation(max_length=2048)

    dataset = load_dataset("web_of_science", 'WOS46985')
    map_func = partial(tokenize, tokenizer)
    tokenized_data = dataset['train'].map(
        map_func,
        num_proc=8,
        batched=True,
        batch_size=512,
        remove_columns=['label', 'input_data', 'label_level_1', 'label_level_2']
    )
    # tokenized_data.save_to_disk('./preprocessed_data/')  # File too large
    ids = tokenized_data['id']
    attention_masks = tokenized_data['attention_mask']
    np.savez_compressed('./processed_data.npz', id=np.asarray(ids), attention_mask=np.asarray(attention_masks))
    print('Saved processed data to ./processed_data.npz')
