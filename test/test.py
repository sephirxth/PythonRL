import torch
import torch.nn as nn

embed_dim=10
num_heads=5
query=key=value=torch.rand([11,5,1])
print(query)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
print("attn_output = ",attn_output)
print("attn_output_weights",attn_output_weights)
print("multihead_attn(query, key, value)",multihead_attn(query, key, value))