import json
import pickle
import pandas as pd
import os
import csv
import hashlib
import os.path as osp
import time
import numpy as np


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj,
                      (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                       np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray, )):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


# LOAD & DUMP
def dump(data, f, **kwargs):

    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data,
                  open(pth, 'w'),
                  indent=4,
                  ensure_ascii=False,
                  cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [
            json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data
        ]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f,
                    sep='\t',
                    index=False,
                    encoding='utf-8',
                    quoting=quoting)

    handlers = dict(pkl=dump_pkl,
                    json=dump_json,
                    jsonl=dump_jsonl,
                    xlsx=dump_xlsx,
                    csv=dump_csv,
                    tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f):

    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl,
                    json=load_json,
                    jsonl=load_jsonl,
                    xlsx=load_xlsx,
                    csv=load_csv,
                    tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f)
