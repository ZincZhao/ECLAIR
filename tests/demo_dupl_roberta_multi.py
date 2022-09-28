
import torch

from transformers import RobertaTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments
from eclare.duplication.extract.duplication_generator import duplication_generator
from eclare.duplication.extract.duplication_generator_expanded import duplication_generator_expanded

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load("/local/scratch/bzhao44/baseline2-models/roberta-baseline2-3")
model = model.to(device)
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

generator = duplication_generator_expanded
sample_dataset = generator.generate('/local/scratch/bzhao44/ECLAIR/res/data_splits_updated.xlsx', '/local/scratch/bzhao44/ECLAIR/res/rchilli/')


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_data = sample_dataset.map(preprocess_function, batched=True)
# tokenized_data = tokenized_data.remove_columns(customized_data["train"].column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="/local/scratch/bzhao44/ECLAIR/res/output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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

prediction = trainer.predict(tokenized_data['test'])

print()

pred_res = []
actl_res = []

for elem in prediction.predictions:
    mx = -100
    mxidx = elem
    for i in range(0, len(elem)):
        if elem[i] > mx:
            mx = elem[i]
            mxidx = i
    if mxidx == 0:
        pred_res.append(0)
    elif mxidx == 1:
        pred_res.append(1)
    elif mxidx == 2:
        pred_res.append(2)
    elif mxidx == 3:
        pred_res.append(3)
    elif mxidx == 4:
        pred_res.append(4)
    elif mxidx == -1:
        print("!!!NOT VALID!!!")
    else:
        print("!!!NOT RECOGNIZED!!!")

for elem in sample_dataset['test']:
    actl_res.append(elem['label'])

counter = 0
correct_counter = 0

confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

for i in range(0, len(pred_res)):
    if pred_res[i] == actl_res[i]:
        correct_counter = correct_counter + 1
        confusion_matrix[pred_res[i]][pred_res[i]] = confusion_matrix[pred_res[i]][pred_res[i]] + 1
    else:
        print("This Resume is predicted incorrectly:" + str(counter))
        print("The Correct res is " + str(actl_res[i]))
        print("The Predicted res is " + str(pred_res[i]))
        print()
        confusion_matrix[actl_res[i]][pred_res[i]] = confusion_matrix[actl_res[i]][pred_res[i]] + 1
    counter = counter + 1


print()
print("ACTL NQ:")
print(confusion_matrix[0])
print("ACTL CRCI:")
print(confusion_matrix[1])
print("ACTL CRCII:")
print(confusion_matrix[2])
print("ACTL CRCIII:")
print(confusion_matrix[3])
print("ACTL CRCIV:")
print(confusion_matrix[4])

print()
print(correct_counter)
print(counter)
