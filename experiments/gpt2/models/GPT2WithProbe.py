# https://chatgpt.com/c/68016f07-f304-8004-a303-fcb4f1e94a9f?model=gpt-4o

import torch.nn as nn
from transformers import GPT2Model

class GPT2WithProbe(nn.Module):
    def __init__(self, base_model: GPT2Model, num_labels=2):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size

        # Example probe: simple MLP classifier head
        self.probe_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape (batch, seq_len, hidden)

        # Let's probe the *final* token (causal GPT style)
        pooled_output = hidden_states[:, -1, :]  # (batch, hidden)
        logits = self.probe_head(pooled_output)  # (batch, num_labels)

        return logits
