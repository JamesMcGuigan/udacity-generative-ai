# GPT2LMHeadModel
# https://chatgpt.com/c/68016f07-f304-8004-a303-fcb4f1e94a9f
#
# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (wte): Embedding(50257, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     (h): ModuleList(
#       (0-11): 12 x GPT2Block(
#         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (attn): GPT2Attention(
#           (c_attn): Conv1D(nf=2304, nx=768)
#           (c_proj): Conv1D(nf=768, nx=768)
#           (attn_dropout): Dropout(p=0.1, inplace=False)
#           (resid_dropout): Dropout(p=0.1, inplace=False)
#         )
#         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (mlp): GPT2MLP(
#           (c_fc): Conv1D(nf=3072, nx=768)
#           (c_proj): Conv1D(nf=768, nx=3072)
#           (act): NewGELUActivation()
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=50257, bias=False)
# )
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Load model & tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model     = GPT2LMHeadModel.from_pretrained(model_name)

# 2. (Optional) Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(
    prompt: str,
    max_length: int = 50,
    num_return_sequences: int = 1,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> list[str]:
    """
    Generates text continuations for a given prompt.

    Args:
        prompt: Input string to start generation.
        max_length: Total token length (prompt + generated).
        num_return_sequences: How many variations to return.
        temperature: Sampling temperature; higher -> more random.
        top_k: Top-K sampling.
        top_p: Nucleus (top-p) sampling.

    Returns:
        A list of generated strings (length = num_return_sequences).
    """
    # Tokenize input and move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generate
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and return
    results = []
    for generated_ids in outputs:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)
    return results

# Example usage
if __name__ == "__main__":
    prompt_text = "Once upon a time"
    continuations = generate_text(prompt_text, max_length=120, num_return_sequences=1)
    for i, cont in enumerate(continuations, 1):
        print(f"=== Generated #{i} ===")
        print(cont)
        print()
