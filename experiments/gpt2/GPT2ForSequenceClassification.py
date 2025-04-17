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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForSequenceClassification, AutoModelForSequenceClassification

# 1. Load model & tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model     = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

# 2. (Optional) Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(
    prompt: str,
    # max_length: int = 50,
    # num_return_sequences: int = 1,
    # temperature: float = 1.0,
    # top_k: int = 50,
    # top_p: float = 0.95,
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
    # input_ids = inputs.input_ids.to(device)
    # attention_mask = inputs.attention_mask.to(device)



    inputs = tokenizer("This is great!", return_tensors="pt", padding=True)
    logits = model(**inputs).logits
    # Convert to probability
    probs = torch.sigmoid(logits)

    # Optional threshold (usually 0.5)
    predicted_class = (probs > 0.5).long()
    predicted_bool  = torch.argmax(predicted_class, dim=-1).item()

    # print(logits)
    # print(probs)
    # print(predicted_class)
    # print(predicted_bool)

    return predicted_bool

# Example usage
if __name__ == "__main__":
    prompt_text = "Once upon a time"
    continuations = generate_text(prompt_text)