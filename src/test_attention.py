# src/test_attention.py
# KESAV runs this on ROG STRIX
# Goal: see what a raw attention matrix looks like from a real model

import torch
from transformers import AutoTokenizer, AutoModel

# Use a tiny model first — just to understand the shape
# This is NOT your final model, just for learning
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "A red car is parked next to a blue building."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

attentions = outputs.attentions  # tuple of tensors, one per layer

print(f"Number of layers: {len(attentions)}")
print(f"Shape of layer 0 attention: {attentions[0].shape}")
# Expected: torch.Size([1, 12, seq_len, seq_len])
# Meaning:  [batch, heads, token_i, token_j]

# Print the attention weights from head 0, layer 0
print("\nLayer 0, Head 0 attention matrix:")
print(attentions[0][0][0])  # [seq_len x seq_len]

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"\nTokens: {tokens}")
