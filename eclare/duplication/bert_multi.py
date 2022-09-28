import torch

# from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, TextClassificationPipeline

from eclare.duplication.extract.duplication_generator import duplication_generator
from eclare.duplication.extract.duplication_generator_expanded import duplication_generator_expanded

generator = duplication_generator_expanded
customized_data = generator.generate('/local/scratch/bzhao44/ECLAIR/res/data_splits_updated.xlsx', '/local/scratch/bzhao44/ECLAIR/res/rchilli/')
tokenizer = BertTokenizer.from_pretrained("bert-large-cased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_data = customized_data.map(preprocess_function, batched=True)
# tokenized_data = tokenized_data.remove_columns(customized_data["train"].column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# don't forget to change the number of labels
model = BertForSequenceClassification.from_pretrained("/local/scratch/bzhao44/ECLAIR/res/output/checkpoint-3000", num_labels=5)
# model=BertForSequenceClassification.from_pretrained("bert-large-cased", num_labels=5)

training_args = TrainingArguments(
    output_dir="/local/scratch/bzhao44/ECLAIR/res/output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
torch.save(model, "/local/scratch/bzhao44/baseline3-models/bert-baseline3-1")
# accuracy = trainer.evaluate()
# print(accuracy)

