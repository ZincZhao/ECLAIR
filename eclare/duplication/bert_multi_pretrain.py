import torch
import pytorch_lightning
import tokenizers
import os
import json
import gc

# from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, BertConfig, TrainingArguments, Trainer, LineByLineTextDataset, TextDataset
from tokenizers import *
from datasets import *

from eclare.duplication.extract.duplication_generator import duplication_generator
from eclare.duplication.extract.duplication_generator_expanded import duplication_generator_expanded
from eclare.duplication.extract.duplication_generator_pretrain import duplication_generator_pretrain

generator = duplication_generator_pretrain
gen_dataset = generator.generate('/local/scratch/bzhao44/ECLAIR/res/data_splits_updated.xlsx', '/local/scratch/bzhao44/ECLAIR/res/rchilli/')

tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

lin_dataset = LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = '/local/scratch/bzhao44/ECLAIR/res/ext_data/dataset.txt',
    block_size = 128
)

def preprocess_function(examples):
    return tokenizer(examples, truncation=True)

gen_dataset = torch.load(lin_dataset)

# gen_dataset = load_dataset("text", data_dir="/local/scratch/bzhao44/ECLAIR/res/ext_data/")


# tokenized_data = gen_dataset.map(preprocess_function, batched=True)
# tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

config = BertConfig(
    vocab_size=50000,
    max_position_embeddings=512,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16
)

model = BertForMaskedLM(config)

training_args = TrainingArguments(
    output_dir="/local/scratch/bzhao44/ECLAIR/res/output",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    save_steps=1000,
    gradient_accumulation_steps=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=gen_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model("/local/scratch/bzhao44/pretrained-model/")
# accuracy = trainer.evaluate()
# print(accuracy)

