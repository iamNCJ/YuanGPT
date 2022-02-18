# YuanGenerativeLM
Generative Language Model Pretrained on Inspur's Yuan Dataset, codebase for ASC22 supercomputing competition

## Project Structure

To simplify experiments on different distributed training frameworks, we decoupled the training code into `config`, `data`, `model` and `trainer` modules.

The idea of this decoupling is inspired by pytorch-lightning, however we decoupled it even further to make it more flexible when integrating with other frameworks.

### `config` Module

We put all hyperparameters and configurations into `config` module for better tracing and logging.

### `data` Module

We directly use `pytorch-lightning.LightningDataModule` since it's interface is well-designed and easy to use.

### `model` Module

Since most distributed training framework need to wrap the model before or after model initialization, and `pytorch-lightning.LightningModule` has already exposed some problem in integrating multiple frameworks simultaneously, we decide to further decouple this module into `BaseModel` class.

The `BaseModel` directly inherits `nn.Module`, which is the compatible for most of the distributed training frameworks. All implementations of the language model are derived from `BaseModel` and maintain only the model config, the model structure, the forward method, the loss function and the optimizer.

Currently, implemented models include:
- native model: written in native pytorch
- huggingface model: written in HuggingFace's transformers

### `trainer` Module

Now we put everything else like model initialization, training, validation and testing into `trainer` module. All training preparation and iterations are done here.

Currently, implemented trainers include:
- PytorchLightning trainer: distributed training with pytorch-lightning, with deepspeed integration provided by the lightning team
- PatrickStar Trainer

## Distributed Launch

Below are examples of how to launch the training job on different distributed frameworks.

### DDP in PyTorch-Lightning

`num_nodes` must be set to number of GPUs in all nodes, otherwise it will use the number of GPUs in the master node.

```sh
torchrun --nnodes=2 --nproc_per_node=2 --master_addr GPU04 --master_port 9001 --node_rank 1 train.ddp_pl.py
```

### DeepSpeed in PyTorch-Lightning

```sh
OMP_NUM_THREADS=32 torchrun --nnodes=2 --nproc_per_node=2 --master_addr GPU04 --master_port 9001 --node_rank 1 train.ds_pl.py
```

Note that `OMP_NUM_THREADS` is a must when offload is used, since Optimizer now runs on CPU. 

### Horovod in PyTorch-Lightning

```sh
horovodrun -np 2 python train.hvd_pl.py
```

We still prefer to use `torchrun`

### PatrickStar

```sh
torchrun --nnodes=1 --nproc_per_node=2 train.pstar.py
```

## Run Profile

```sh
nvprof --profile-from-start off -o xxx.nvprof -- OMP_NUM_THREADS=32 torchrun --nnodes=2 --nproc_per_node=2 --master_addr GPU04 --master_port 9001 --node_rank 1 train.ds_pl.py
```

## Docker Environment

```sh
docker run -it --name pytorch --gpus all --privileged --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/infiniband -v $(pwd):/workspace registry.cn-hangzhou.aliyuncs.com/ncj/pytorch bash
```

Check details in [Dockerfile](./Dockerfile)