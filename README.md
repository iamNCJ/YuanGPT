# YuanGenerativeLM
Generative Language Model Pretrained on Inspur's Yuan Dataset

## Distributed Launch

### DDP in PyTorch-Lightning

`num_nodes` must be set to number of GPUs in all nodes, otherwise it will use the number of GPUs in the master node.

```sh
torchrun --nnodes=2 --nproc_per_node=2 --master_addr GPU04 --master_port 9001 --node_rank 1 train.ddp_pl.py
```

### Horovod in PyTorch-Lightning

```sh
horovodrun -np 2 python train.hvd_pl.py
```

We still prefer to use DDP
