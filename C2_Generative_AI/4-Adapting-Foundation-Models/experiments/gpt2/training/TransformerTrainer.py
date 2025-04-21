# https://learn.udacity.com/nanodegrees/nd608/parts/cd13303/lessons/ab801cf5-1646-4b3b-8891-5959a69a53ea/concepts/f758abc2-06ab-44cb-8823-02b8b308ea4a?lesson_tab=lesson

import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
def train(model, tokenizer, tokenized_dataset):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./data/spam_not_spam",
            learning_rate=2e-3,
            # Reduce the batch size if you don't have enough memory
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        ),
        train_dataset   = tokenized_dataset["train"],
        eval_dataset    = tokenized_dataset["test"],
        tokenizer       = tokenizer,
        data_collator   = DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics = compute_metrics,
    )
    trainer.train()