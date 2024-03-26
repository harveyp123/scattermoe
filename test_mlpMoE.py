import torch
import torch.nn as nn
import torch.nn.functional as F
from scattermoe.mlp import MLP, MLPSparseMoe, GLUMLPSparseMoe

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

GLUMLPMoe = GLUMLPSparseMoe(input_size=x_dim, hidden_size=h_dim,
    num_experts=E, top_k=k)

X = torch.randn(n_input, x_dim).cuda()


# Calling module...
Y, router_logits = mlpMoe(
    X
)

print(Y.shape)
print(router_logits.shape)

# Calling module...
Y2, router_logits2 = GLUMLPMoe(
    X
)

print(Y2.shape)
print(router_logits2.shape)


