from datasets import load_dataset

from eclare.duplication.extract.duplication_generator import duplication_generator

generator = duplication_generator
compare_dd = load_dataset("imdb")
dd = generator.generate('../res/data_splits.xlsx', '../res/rchilli/')
print(dd)
print(compare_dd['test'][0])
print(compare_dd)
