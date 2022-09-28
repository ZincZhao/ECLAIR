import os

import datasets
import pandas
import torch

from eclare.duplication.extract.duplication_extractor_pretrain import duplication_extractor_pretrain
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class LegalDataset(Dataset):
    def __init__(self, text):
        self.encodings = text

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        item = {"input_ids": torch.tensor(self.encodings.iloc[index])}
        return item


class duplication_generator_pretrain:
    def generate(s: str, dir: str) -> datasets.DatasetDict:
        extractor = duplication_extractor_pretrain

        df_dev = pandas.read_excel(s, usecols='A, C', sheet_name=1)
        df_dev = df_dev.to_dict('split')
        df_dev = df_dev['data']

        df_tst = pandas.read_excel(s, usecols='A, C', sheet_name=2)
        df_tst = df_tst.to_dict('split')
        df_tst = df_tst['data']

        text = []

        for f in os.listdir(dir):
            if not f.endswith('.json'):
                continue
            res = extractor.extract(dir + f)
            name = f[:-5]
            found = False
            for elem in df_tst:
                if elem[0] == int(name):
                    found = True
            for elem in df_dev:
                if elem[0] == int(name):
                    found = True
            if not found:
                for i in res:
                    text.append(i)
        
        with open('/local/scratch/bzhao44/ECLAIR/res/ext_data/dataset.txt', 'w') as f:
            for elem in text:
                f.write(elem)
                f.write('\n')

        return text

    def process_text(filename, name, map_tokenize, encoding):
        print("Opening file...")
        file = open(filename, "r", encoding=encoding)
        text = file.readlines() # list
        file.close()
        text = pandas.Series(text)
        tqdm.pandas(desc="Tokenizing")
        text = text.progress_map(map_tokenize)
        dataset = LegalDataset(text)
        text = None
        occ = filename.rfind("/") + 1
        path = filename[:occ]
        torch.save(dataset, path+name+".pt")
        return path+name+".pt"
