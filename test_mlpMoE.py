import torch
import torch.nn as nn
import torch.nn.functional as F
from scattermoe.mlp import MLP, MLPSparseMoe

n_input = 50  #### Number of input
x_dim = 2048  #### Input didden dimension
h_dim = 2048  #### Hidden dimension
E = 8 #### Number of expert
k = 2 #### top k k value

mlpMoe = MLPSparseMoe(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
).cuda()



X = torch.randn(n_input, x_dim).cuda()


# Calling module...
Y, router_logits = mlpMoe(
    X
)

print(Y.shape)
print(router_logits.shape)