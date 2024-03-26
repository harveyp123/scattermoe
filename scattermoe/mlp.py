import torch
from torch import nn
from torch.nn import functional as F

from . import kernels
from .parallel_experts import ParallelExperts

class GLUMLP(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        num_experts,
        top_k,
        activation=nn.SiLU(),
    ):
        super(GLUMLP, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, 2 * hidden_size)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(expert_idxs)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_experts)

        h, gates  = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_out=True
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True,
            gates=expert_p,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y

class GLUMLPSparseMoe(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        num_experts,
        top_k,
        activation=nn.SiLU(),
    ):
        super(GLUMLPSparseMoe, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, 2 * hidden_size)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)


    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor):

        #### Compute router
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)


        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(selected_experts)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_experts)

        h, gates  = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_out=True
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True,
            gates=routing_weights,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y, router_logits

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        activation=None,
    ):
        super(MLP, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, hidden_size)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(expert_idxs)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_experts)

        h = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_out=True
        )
        h = self.activation(h)
        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True,
            gates=expert_p,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y


class MLPSparseMoe(nn.Module):
    #### Following: https://github.com/harveyp123/scattermoe/blob/b72f826c411f26777d8149d852e63a4822d58d8f/examples/mixtral/modeling_mixtral.py#L667
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        activation=None,
    ):
        super(MLPSparseMoe, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, hidden_size)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor):

        #### Compute router
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)


        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(selected_experts)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_experts)

        h = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_out=True
        )
        h = self.activation(h)
        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True,
            gates=routing_weights,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y, router_logits


class MLPMoe_baseline(nn.Module):
    #### Following: https://github.com/harveyp123/scattermoe/blob/b72f826c411f26777d8149d852e63a4822d58d8f/examples/mixtral/modeling_mixtral.py#L667
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        activation=None,
    ):
        super(MLPMoe_baseline, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.experts = ParallelExperts(num_experts, input_size, hidden_size)
        # self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.experts = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size) for _ in range(self.num_experts)])
        self.output_experts = nn.ModuleList([nn.Linear(self.hidden_size, self.input_size) for _ in range(self.num_experts)])
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor):

        #### Compute router
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)


        x_shape = x.size()

        final_hidden_states = torch.zeros(x.shape, dtype=x.dtype, device=x.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)


        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[top_x_list]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))


        return final_hidden_states, router_logits