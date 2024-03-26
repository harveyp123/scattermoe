import torch
import torch.nn as nn
import torch.nn.functional as F
from scattermoe.mlp import MLP

n_input = 50  #### Number of input
x_dim = 2048  #### Input didden dimension
h_dim = 2048  #### Hidden dimension
E = 8 #### Number of expert
k = 2 #### top k k value


class MoERouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super(MoERouter, self).__init__()
        # Assuming a simple MLP for routing with one hidden layer
        self.fc1 = nn.Linear(d_model, num_experts * 2)
        self.fc2 = nn.Linear(num_experts * 2, num_experts)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


# Initialise module...
mlp = MLP(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
)

# Initialise router...
router = MoERouter(d_model = x_dim, num_experts = E).cuda()

mlp = MLP(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
).cuda()


X = torch.randn(n_input, x_dim).cuda()
router_weights = router(X)

### Following router writing from https://github.com/huggingface/transformers/blob/092f1fdaa4224fdd88c616dc9678e6fcb37bfffd/src/transformers/models/mixtral/modeling_mixtral.py#L847
router_weights, selected_experts = torch.topk(router_weights, k, dim=-1)
router_weights /= router_weights.sum(dim=-1, keepdim=True)


# Calling module...
Y = mlp(
    X,         # input tensor
    router_weights, # top-k weights from router
    selected_experts     # top-k indices from router
)

print(Y)