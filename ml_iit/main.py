from transformers import (
    AutoTokenizer,
    XLMRobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
import evaluate
import torch
from numba import cuda as cudaa
devicew = cudaa.get_current_device()

torch.cuda.empty_cache()
import gc
gc.collect()
print(torch.cuda.memory_summary(device=None, abbreviated=False))

metric = evaluate.load("accuracy")

model_checkpoint = "FacebookAI/xlm-roberta-base"
model_name = model_checkpoint.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
def model_init():
  devicew.reset()
  torch.cuda.empty_cache()
  return XLMRobertaForSequenceClassification.from_pretrained(
		model_checkpoint, num_labels=5
  )
dataset = load_dataset("/content/drive/MyDrive/dataset")
#print(dataset)

def encode(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


encoded_dataset = dataset.map(encode, batched=True)
batch_size = 8
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=64,
    push_to_hub=True,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)



trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
best_run
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)
trainer.train()
trainer.evaluate()