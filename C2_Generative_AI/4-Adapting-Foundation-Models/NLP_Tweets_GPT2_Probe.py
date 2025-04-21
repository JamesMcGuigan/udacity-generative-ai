# NLP Tweets - GPT2 Probe
# https://www.kaggle.com/code/jamesmcguigan/nlp-tweets-gpt2-probe

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Model

model_name = "gpt2"
model      = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.pad_token_id = model.config.eos_token_id
model.to(device)

tokenizer  = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})


## Parameter Efficient Fine-Tuning - only train probe head, not base model
for param in model.parameters():            param.requires_grad = True
for param in model.base_model.parameters(): param.requires_grad = False



# Dataframes
## !ls -la /kaggle/input/nlp-getting-started

from sklearn.model_selection import train_test_split

train_df_csv         = pd.read_csv('./nlp-getting-started/train.csv')
test_df              = pd.read_csv('./nlp-getting-started/test.csv')
sample_submission_df = pd.read_csv('./nlp-getting-started/sample_submission.csv')

train_df, eval_df = train_test_split(train_df_csv, test_size=0.2, stratify=train_df_csv['target'], random_state=42)



# Huggingface Datasets

from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(train_df)
eval_dataset  = Dataset.from_pandas(eval_df)
test_dataset  = Dataset.from_pandas(test_df)

## Tokenize in batch mode (recommended)
def tokenize_function(batch):
    tokenized = tokenizer(batch["text"], padding="max_length", truncation=True)
    if 'target' in batch:
        tokenized['labels'] = batch['target']
        # tokenized["labels"] = [float(x) for x in batch["target"]] # Convert ints to floats for MSELoss
    return tokenized

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval  =  eval_dataset.map(tokenize_function, batched=True)
tokenized_test  =  test_dataset.map(tokenize_function, batched=True)

tokenized_dataset = DatasetDict({
    "train": tokenized_train,
    "eval":  tokenized_eval,
    "test":  tokenized_test,
})


# %%time
# LINT-FIX: You're using a DistilBertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/sentiment_analysis",
        learning_rate=2e-3,
        per_device_train_batch_size=16,  # Reduce the batch size if you don't have enough memory
        per_device_eval_batch_size=16,
        num_train_epochs=10 if (os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Batch') else 1,
        weight_decay=0.01,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        report_to="none"
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("START trainer.train()")
trainer.train()
print("END   trainer.train()")


# Submission.csv

# %%time
output = trainer.predict(tokenized_test)
logits = output.predictions        # shape (n_samples, num_labels) or (n_samples,)
preds  = np.argmax(logits, axis=1) # 4. Convert logits → {0,1} targets If you used a 2‑class head (num_labels=2):

submission = pd.DataFrame({
    "id":     tokenized_dataset["test"]["id"],
    "target": preds
})

submission.to_csv("submission.csv", index=False)