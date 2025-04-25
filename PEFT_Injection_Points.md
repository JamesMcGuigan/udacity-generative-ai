# PEFT Injection Points
- https://chatgpt.com/c/680609fe-cbd4-8004-a8c2-a069a1e57741

Below is a walk‑through of the GPT‑2 sequence‑classification model’s core parts, and two patterns for “injecting” trainable layers into a frozen foundation model:  

---

## 1. Anatomy of `GPT2ForSequenceClassification`

```
GPT2ForSequenceClassification(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)           # token embeddings
    (wpe): Embedding(1024, 768)            # positional embeddings
    (drop): Dropout(p=0.1)                 # input dropout
    (h): ModuleList([                      # 12× GPT2Block
      GPT2Block(
        (ln_1): LayerNorm(768)             # pre‐attention layer norm
        (attn): GPT2Attention(             # multi‑head self‑attention
          (c_attn): Conv1D(2304, 768)      # QKV projection
          (c_proj): Conv1D(768, 768)       # output projection
          (attn_dropout): Dropout(0.1)
          (resid_dropout): Dropout(0.1)
        )
        (ln_2): LayerNorm(768)             # post‑attention layer norm
        (mlp): GPT2MLP(                    # feed‑forward MLP
          (c_fc): Conv1D(3072, 768)        # first FC
          (c_proj): Conv1D(768, 3072)      # second FC
          (act): NewGELUActivation()       # GELU
          (dropout): Dropout(0.1)
        )
      )
      …  # repeated 12 times
    ])
    (ln_f): LayerNorm(768)                 # final layer norm
  )
  (score): Linear(768 → num_labels)        # your classification head
)
```

- **Embeddings (wte, wpe)** produce the 768‑dim “hidden states.”  
- **Dropout (drop)** regularizes early.  
- **Each GPT2Block**:  
  - `ln_1` → self‑attention → `ln_2` → MLP → (residuals everywhere).  
- **ln_f** normalizes the final hidden state.  
- **score** is your linear probe head (frozen base + trainable `score`).  

In your snippet you do:
```python
# freeze foundation
for p in model.base_model.parameters():
    p.requires_grad = False

# leave `model.score` trainable
```

---

## 2. Pattern A: Manually Add “Adapter” Modules

You can slip small trainable “adapter” layers inside each block:

```python
import torch, torch.nn as nn
from transformers import GPT2ForSequenceClassification

# 1) Load & freeze
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
for p in model.base_model.parameters():
    p.requires_grad = False

# 2) Define a tiny adapter
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, dim)
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # residual

# 3) Inject into each transformer block
for block in model.base_model.h:
    block.adapter = Adapter(model.config.hidden_size)

# 4) Monkey‑patch block forward
_original_forward = type(model.base_model.h[0]).forward
def forward_with_adapter(self, hidden_states, *args, **kwargs):
    hidden_states, presents = _original_forward(self, hidden_states, *args, **kwargs)
    hidden_states = self.adapter(hidden_states)
    return hidden_states, presents

type(model.base_model.h[0]).forward = forward_with_adapter
```

- **What happens**: Each block now applies your tiny `Adapter` after the MLP, and only those adapter params (plus the final `score` head) are updated.  
- **Pros**: Very small parameter overhead (~2× bottleneck×hidden_size per block).  
- **Cons**: Monkey‑patching can get brittle; be sure your custom forward signature matches exactly.

---

## 3. Pattern B: Use a PEFT Library (LoRA / Adapters)

The [PEFT](https://github.com/huggingface/peft) library makes this injection seamless:

```python
from peft import get_peft_model, LoraConfig

# 1) Load & freeze (PEFT will handle freezing automatically)
base_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

# 2) Configure LoRA—only inject trainable rank‑r matrices into attention & MLP
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn","c_proj"],  # you can also target "mlp.c_fc", etc.
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

# 3) Wrap model
model = get_peft_model(base_model, lora_cfg)
```

- **Under the hood**:  
  - PEFT freezes all original weights,  
  - injects low‑rank A/B matrices (LoRA) into your specified modules,  
  - handles training only those new weights.  
- **Syntax** is just a few lines—no manual forward‑patching.

---

## 4. Summary & Next Steps

- **Freezing**: `for p in model.base_model.parameters(): p.requires_grad=False`.  
- **Manual adapters**: define small `nn.Module`, attach to each block, monkey‑patch `forward`.  
- **PEFT / LoRA**: install `peft` and call `get_peft_model`. Super‑lightweight and less error‑prone.  

---

❓ **Questions to refine further**  
1. Do you want an end‑to‑end example showing training code with PEFT?  
2. Interested in other injection points (e.g. after attention vs after MLP)?  
3. Planning to compare adapter vs LoRA vs full head‑only fine‑tuning on your dataset?  

Let me know which angle you’d like to deep‑dive!


# Other Injection Points

Great—let’s map out the key “slots” where you can slip in trainable modules, and show you how to hook into each one. In GPT‑2’s `TransformerBlock`, the rough flow is:

```
  x ──► ln_1 ──► Self‑Attention ──► +res ► ln_2 ──► MLP ──► +res ──► output
                │                            │
                │                            └──► Final LayerNorm (ln_f)
                └──► (projection layers inside Attn & MLP)
```

Below are the main injection points—with code sketches—for adapters, prefix‑tokens, or LoRA, plus trade‑offs.

---

## 1. After Self‑Attention (Post‑Attn Adapter)

**Why?**  
– Captures “what the model just attended to,” letting you reshape or gate attention outputs before they mix into the feed‑forward path.  
– Often a good balance of expressivity vs. size.

```python
import torch.nn as nn

# 1) Freeze foundation
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
for p in model.base_model.parameters():
    p.requires_grad = False

# 2) Define a small adapter
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, dim)
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

# 3) Attach & patch forward of each block
for block in model.base_model.h:
    block.attn_adapter = Adapter(model.config.hidden_size)
_original_attn_forward = type(model.base_model.h[0]).attn.forward
def attn_forward_with_adapter(self, hidden_states, *args, **kwargs):
    # run original self‐attention
    attn_output, present = _original_attn_forward(self, hidden_states, *args, **kwargs)
    # apply adapter
    attn_output = self.attn_adapter(attn_output)
    return attn_output, present

# Monkey‑patch the attention module
type(model.base_model.h[0]).attn.forward = attn_forward_with_adapter
```

---

## 2. After MLP (Post‑FFN Adapter)

**Why?**  
– Lets you reshape the feed‑forward features, which often carry deeper semantic information.  
– Tends to give a bit more capacity than post‑attn for a similar bottleneck size.

```python
# 1) Freeze foundation (same as above)

# 2) Attach & patch MLP
for block in model.base_model.h:
    block.ffn_adapter = Adapter(model.config.hidden_size)

_original_block_forward = type(model.base_model.h[0]).forward
def block_forward_with_ffn_adapter(self, hidden_states, *args, **kwargs):
    h, presents = _original_block_forward(self, hidden_states, *args, **kwargs)
    h = self.ffn_adapter(h)
    return h, presents

# Monkey‑patch the whole block
type(model.base_model.h[0]).forward = block_forward_with_ffn_adapter
```

---

## 3. Inside the Attention (LoRA on Q, K, V or Proj)

With **LoRA**, you inject low‑rank updates directly into the projection matrices:

```python
from peft import get_peft_model, LoraConfig

base = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
# Config: inject into QKV and/or output proj
lora_cfg = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["c_attn", "c_proj"],  # "c_attn" covers Q, K, V
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(base, lora_cfg)
```

- **`target_modules=["c_attn"]`**: injects into the combined QKV.  
- **`target_modules=["attn.c_proj"]`**: only into the output projection of attention.  

---

## 4. Prefix‑Tuning / Prompt‑Tuning

Rather than adapters, you can learn a small set of “virtual tokens” prepended to each layer’s key/value:

```python
from peft import PrefixTuningConfig, get_peft_model

base = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
pt_cfg = PrefixTuningConfig(
    task_type="SEQ_CLS",
    prefix_length=20,           # learn 20 virtual tokens per layer
    num_virtual_tokens=20,
    target_modules=["h"],       # inject into all transformer blocks
)
model = get_peft_model(base, pt_cfg)
```

- Virtual tokens become additional context for every attention head.  
- Parameter count = `num_layers × prefix_length × hidden_size`.

---

## 5. Trade‑Offs

| Injection Point       | Params Overhead     | Expressivity         | Typical Use                         |
|-----------------------|---------------------|----------------------|-------------------------------------|
| **Post‑Attn Adapter** | ~2×64×768/block     | Medium               | Fast prototyping, stable training   |
| **Post‑FFN Adapter**  | ~2×64×768/block     | Medium–High          | Better for semantic FFN tweaks      |
| **LoRA on QKV**       | ~2×r×(3×768)        | High (attn‑specific) | Fine‑grained control of attention   |
| **LoRA on c_proj**    | ~2×r×768            | Low–Medium           | Lightweight, quick wins             |
| **Prefix‑Tuning**     | L×P×D               | High (contextual)    | When full attention control needed  |

---

### 🧠 Next Questions

1. **Which trade‑off matters to you most?** (size vs. expressivity vs. simplicity)  
2. **Want a script that runs side‑by‑side benchmarks** (e.g., train adapters at post‑attn vs. post‑FFN)?  
3. **Curious about more exotic slots** (e.g., right after embeddings, or before the final `score` head)?  

— and as a tip for future prompting: if you know in advance you want to compare two or more schemes, your prompt could be  
> “Generate me training/evaluation code that simultaneously sets up post‑attn adapters and post‑FFN adapters, then logs their eval accuracies on the same split.”