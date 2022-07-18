from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

from eclare.duplication.extract.duplication_generator import duplication_generator

generator = duplication_generator
customized_data = generator.generate('../../res/data_splits.xlsx', '../../res/rchilli/')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_data = customized_data.map(preprocess_function, batched=True)
# tokenized_data = tokenized_data.remove_columns(customized_data["train"].column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# don't forget to change the number of labels
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

training_args = TrainingArguments(
    output_dir="../../res/output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
