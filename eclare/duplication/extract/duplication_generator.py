import os

import datasets
import pandas

from eclare.duplication.extract.duplication_extractor import duplication_extractor


class duplication_generator:
    def generate(s: str, dir: str) -> datasets.DatasetDict:
        extractor = duplication_extractor

        df_trn = pandas.read_excel(s, usecols='A, C')
        # df_trn['data'].replace(['NQ', 'CRCI', 'CRCII', 'CRCIII', 'CRCIV'], [0, 1, 2, 3, 4])
        df_trn = df_trn.to_dict('split')
        df_trn = df_trn['data']

        df_tst = pandas.read_excel(s, usecols='A, C', sheet_name=2)
        # df_tst['data'].replace(['NQ', 'CRCI', 'CRCII', 'CRCIII', 'CRCIV'], [0, 1, 2, 3, 4])
        df_tst = df_tst.to_dict('split')
        df_tst = df_tst['data']

        ls_trn = {}
        ls_tst = {}
        lb_trn = []
        lb_tst = []
        txt_trn = []
        txt_tst = []

        for f in os.listdir(dir):
            if not f.endswith('.json'):
                continue
            res = extractor.extract(dir + f)
            name = f[:-5]
            for elem in df_trn:
                if elem[0] == int(name):
                    lb_trn.append(elem[1])
                    txt_trn.append(res)
            for elem in df_tst:
                if elem[0] == int(name):
                    lb_tst.append(elem[1])
                    txt_tst.append(res)

        ls_trn['text'] = txt_trn
        ls_trn['label'] = []
        for elem in lb_trn:
            if elem == 'NQ':
                ls_trn['label'].append(0)
            elif elem == 'CRCI':
                ls_trn['label'].append(1)
            elif elem == 'CRCII':
                ls_trn['label'].append(2)
            elif elem == 'CRCIII':
                ls_trn['label'].append(3)
            elif elem == 'CRCIV':
                ls_trn['label'].append(4)
            else:
                ls_trn['label'].append(100)

        ls_tst['text'] = txt_tst
        ls_tst['label'] = []
        for elem in lb_tst:
            if elem == 'NQ':
                ls_tst['label'].append(0)
            elif elem == 'CRCI':
                ls_tst['label'].append(1)
            elif elem == 'CRCII':
                ls_tst['label'].append(2)
            elif elem == 'CRCIII':
                ls_tst['label'].append(3)
            elif elem == 'CRCIV':
                ls_tst['label'].append(4)
            else:
                ls_tst['label'].append(100)

        trn_set = datasets.Dataset.from_dict(ls_trn)
        tst_set = datasets.Dataset.from_dict(ls_tst)
        dd = datasets.DatasetDict({"train": trn_set, "test": tst_set})
        return dd
