from datasets import load_dataset

from eclare.duplication.extract.duplication_generator import duplication_generator
from eclare.duplication.extract.duplication_generator_expanded import duplication_generator_expanded

generator = duplication_generator_expanded

dd = generator.generate('/local/scratch/bzhao44/ECLAIR/res/data_splits_updated.xlsx', '/local/scratch/bzhao44/ECLAIR/res/rchilli/')

print('test starts')
counter = 0
valid_labels = [0, 1, 2, 3, 4]
for elem in dd['dev']:
    print(type(elem))
    print(elem)
print('test ends')
