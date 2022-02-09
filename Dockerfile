FROM nvcr.io/nvidia/pytorch:21.10-py3

RUN pip install pytorch-lightning transformers datasets torchtyping tqdm jieba deepspeed && pip uninstall -y torchtext
