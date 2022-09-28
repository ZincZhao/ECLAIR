import os

import datasets
import pandas

from eclare.duplication.extract.duplication_extractor import duplication_extractor


class duplication_generator_expanded:
    def generate(s: str, dir: str) -> datasets.DatasetDict:
        extractor = duplication_extractor

        df_trn = pandas.read_excel(s, usecols='A, C', sheet_name=[0, 3])
        # df_trn['data'].replace(['NQ', 'CRCI', 'CRCII', 'CRCIII', 'CRCIV'], [0, 1, 2, 3, 4])
        df_trn_aux = []
        for elem in df_trn:
            dc = df_trn[elem].to_dict('split')['data']
            for e in dc:
                df_trn_aux.append(e)
        df_trn = df_trn_aux

        df_dev = pandas.read_excel(s, usecols='A, C', sheet_name=1)
        df_dev = df_dev.to_dict('split')
        df_dev = df_dev['data']

        df_tst = pandas.read_excel(s, usecols='A, C', sheet_name=2)
        # df_tst['data'].replace(['NQ', 'CRCI', 'CRCII', 'CRCIII', 'CRCIV'], [0, 1, 2, 3, 4])
        df_tst = df_tst.to_dict('split')
        df_tst = df_tst['data']

        ls_trn = {}
        ls_dev = {}
        ls_tst = {}
        lb_trn = []
        lb_dev = []
        lb_tst = []
        txt_trn = []
        txt_dev = []
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
            for elem in df_dev:
                if elem[0] == int(name):
                    lb_dev.append(elem[1])
                    txt_dev.append(res)

        ls_trn['text'] = txt_trn
        ls_trn['label'] = []
        for elem in lb_trn:
            if 'NQ' in elem:
                ls_trn['label'].append(0)
            elif 'CRCIII' in elem:
                ls_trn['label'].append(3)
            elif 'CRCII' in elem:
                ls_trn['label'].append(2)
            elif 'CRCIV' in elem:
                ls_trn['label'].append(4)
            elif 'CRCI' in elem:
                ls_trn['label'].append(1)
            else:
                ls_trn['label'].append(100)
        
        ls_dev['text'] = txt_dev
        ls_dev['label'] = []
        for elem in lb_dev:
            if 'NQ' in elem:
                ls_dev['label'].append(0)
            elif 'CRCIII' in elem:
                ls_dev['label'].append(3)
            elif 'CRCII' in elem:
                ls_dev['label'].append(2)
            elif 'CRCIV' in elem:
                ls_dev['label'].append(4)
            elif 'CRCI' in elem:
                ls_dev['label'].append(1)
            else:
                ls_dev['label'].append(100)

        ls_tst['text'] = txt_tst
        ls_tst['label'] = []
        
        for elem in lb_tst:
            if 'NQ' in elem:
                ls_tst['label'].append(0)
            elif 'CRCIII' in elem:
                ls_tst['label'].append(3)
            elif 'CRCII' in elem:
                ls_tst['label'].append(2)
            elif 'CRCIV' in elem:
                ls_tst['label'].append(4)
            elif 'CRCI' in elem:
                ls_tst['label'].append(1)
            else:
                ls_tst['label'].append(100)

        trn_set = datasets.Dataset.from_dict(ls_trn)
        tst_set = datasets.Dataset.from_dict(ls_tst)
        dev_set = datasets.Dataset.from_dict(ls_dev)
        dd = datasets.DatasetDict({"train": trn_set, "dev": dev_set, "test": tst_set})
        return dd
