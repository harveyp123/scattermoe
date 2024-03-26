import torch
import torch.nn as nn
import torch.nn.functional as F
from scattermoe.mlp import MLP, MLPSparseMoe, GLUMLPSparseMoe, MLPMoe_baseline
import time

n_input = 50  #### Number of input
x_dim = 2048  #### Input didden dimension
h_dim = 2048  #### Hidden dimension
E = 8 #### Number of expert
k = 2 #### top k k value
repeats=20 #### Repeat of benchmark
mlpMoe = MLPSparseMoe(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
).cuda()

mlpMoe_baseline = MLPMoe_baseline(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
).cuda()

X = torch.randn(n_input, x_dim).cuda()


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(repeats):
    Y, router_logits = mlpMoe(
        X
    )
end.record()
torch.cuda.synchronize()
print(
    f"Sparse MoE single forward speed (averaged on {repeats} times): {(start.elapsed_time(end)) / repeats}ms"
)

time_start=time.time()
for i in range(repeats):
    torch.cuda.synchronize()
    # Calling module...
    Y, router_logits = mlpMoe(
        X
    )
    # Perform a CUDA operation
    torch.cuda.synchronize()

end_time = time.time()
# Print the elapsed time
# print(event.elapsed_time(event))

# print("avegrage time: {}".format(event.elapsed_time(event)/repeats))

print(f'time used for each forward (sparse MoE): {(end_time-time_start)/repeats}')

print(Y.shape)
print(router_logits.shape)




time_start=time.time()
for i in range(repeats):
    torch.cuda.synchronize()
    # Calling module...
    Y, router_logits = mlpMoe_baseline(
        X
    )
    # Perform a CUDA operation
    torch.cuda.synchronize()

end_time = time.time()
# Print the elapsed time
# print(event.elapsed_time(event))

# print("avegrage time: {}".format(event.elapsed_time(event)/repeats))

print(f'time used for each forward (baseline): {(end_time-time_start)/repeats}')

print(Y.shape)
print(router_logits.shape)

