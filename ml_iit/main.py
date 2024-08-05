# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from pathlib import Path
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=True)
model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=4)
load_dataset("./dataset")
