import sys
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import json
import re
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from huggingface_hub import hf_hub_download

import argparse
import os
import torch.nn.functional as F
import seaborn as sns
import copy

import time

# model_name = "mlabonne/phixtral-2x2_8"

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     device_map="auto"
# )
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_name,
#     padding_side="right",
#     use_fast=False,
# )


model_name = "phixtral-2x2_8"
torch.set_default_device("cuda")
# load_in_4bit=True, 
# Load the model and tokenizer
# device_map="auto"

model = AutoModelForCausalLM.from_pretrained(
    f"mlabonne/{model_name}", 
    torch_dtype="auto", 
    trust_remote_code=True
)

PhiConfig = {
  "_name_or_path": "mlabonne/phixtral-2x2_8",
  "activation_function": "gelu_new",
  "architectures": [
    "PhiForCausalLM"
  ],
  "attn_pdrop": 0.0,
  "auto_map": {
    "AutoConfig": "mlabonne/phixtral-2x2_8--configuration_phi.PhiConfig",
    "AutoModelForCausalLM": "mlabonne/phixtral-2x2_8--modeling_phi.PhiForCausalLM"
  },
  "embd_pdrop": 0.0,
  "flash_attn": false,
  "flash_rotary": false,
  "fused_dense": false,
  "img_processor": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "phi-msft",
  "n_embd": 2560,
  "n_head": 32,
  "n_head_kv": null,
  "n_inner": null,
  "n_layer": 32,
  "n_positions": 2048,
  "num_experts_per_tok": 1,
  "num_local_experts": 4,
  "resid_pdrop": 0.1,
  "rotary_dim": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.39.1",
  "vocab_size": 51200
}

# model = AutoModelForCausalLM.from_config(
#     f"mlabonne/{model_name}", 
#     torch_dtype="auto", 
#     trust_remote_code=True
# )

tokenizer = AutoTokenizer.from_pretrained(
    f"mlabonne/{model_name}", 
    trust_remote_code=True
)

print(model)

instruction = '''
    def print_prime(n):
        """
        Print all primes between 1 and n
        """
'''

# Tokenize the input string
inputs = tokenizer(
    instruction, 
    return_tensors="pt", 
    return_attention_mask=False
)


outputs = model(inputs)



# # Create a CUDA event
# event = torch.cuda.Event(enable_timing=True)
 
# # Start the event
# event.record()
 
# # Perform a CUDA operation
# torch.cuda.synchronize()
 
# # Stop the event
# event.record()
 
# # Print the elapsed time
# print(event.elapsed_time(event))
