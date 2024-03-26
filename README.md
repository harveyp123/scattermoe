# [scattermoe](https://github.com/shawntan/scattermoe)
Triton-based implementation of Sparse Mixture-of-Experts (SMoE) on GPUs.
ScatterMoE builds upon existing implementations, and overcoming some of the limitations to improve inference, training speed, and memory footprint. 
This implementation achieves this by avoiding padding and making excessive copies of the input.
We also fuse expert linear transforms and reordering operations with `ParallelLinear`, a module that can be used to extend the concept of SMoEs.

This implementation is lightweight (~700 lines).
It will work within an FSDP or pipeline parallel framework, but does not include any additional multi-node training infrastructure code.
You can find the report [here](https://arxiv.org/abs/2403.08245)

## Installation
```sh
# Check all is working well.
PYTHONPATH=. pytest tests
# Install editable. This will allow you to modify scattermoe in this directory.
pip install -e .
```

My own environment setup:

```sh
conda create -n "scatter_MoE" python=3.11
conda activate scatter_MoE
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers==4.39.1
pip install chardet==5.2.0
pip install flash-attn==2.5.6
pip install sentencepiece==0.2.0
```



```sh
# Check all is working well.
PYTHONPATH=. pytest tests
# Install editable. This will allow you to modify scattermoe in this directory.
pip install -e .
```

## Finetune
[fastchat Conversation template](https://github.com/leeyeehoo/SleepyLlama/blob/946d70bc7c1587af0b4cc7cfa2b2cf6327deb88b/SleepyLlamaChat/test_sleepy_llama_v2_dreamaug.py#L25C1-L25C65)


## Usage
```python
from scattermoe.mlp import MLP

# Initialise module...
mlp = MLP(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
)

# Calling module...
Y = mlp(
    X,         # input tensor
    k_weights, # top-k weights from router
    k_idxs     # top-k indices from router
)
```

Enjoy!

-----
### More examples
1. [Integration into HuggingFace Mixtral](https://github.com/shawntan/scattermoe/tree/main/examples/mixtral)
