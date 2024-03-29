import os
import os.path as osp
import re
import sys
import math
import tqdm
import time
import json

import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

import torch
from torch.utils.data import Dataset
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)

from ..utils import load


@master_only
def master_print(msg):
    print(msg)


class GeoSceneDataset(Dataset):

    ABBRS = {}

    def __init__(self, name, data_file, image_folder):
        self.name = name
        self.data_file = data_file
        self.image_folder = image_folder
        self.df = load(data_file)
        self.split = 'dev' if 'answer' in self.df[0].keys() else 'test'

    def load_image(self, image_file):
        if image_file.startswith('http://') or image_file.startswith(
                'https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        line = self.df[idx]
        question_id = line['question_id']
        image_file = osp.join(self.image_folder, line['image_path'])
        image = self.load_image(image_file)
        question = line['question']
        answer = self.df[idx]['answer'] if 'answer' in self.df[0].keys(
        ) else None

        data = {
            'question_id': question_id,
            'question': question,
            'answer': answer,
            'image': image,
            'image_path': image_file
        }

        return data

    @master_only
    def eval_result(self, results_df, show=True):

        def calc_acc(df, group='category'):
            assert group in ['overall', 'category']
            if group == 'overall':
                res = {'Average': np.mean(df['hit'])}
            else:
                res = {}
                abilities = list(set(df[group]))
                abilities.sort()
                for ab in abilities:
                    sub_df = df[df[group] == ab]
                    ab = self.ABBRS[ab] if ab in self.ABBRS else ab
                    res[ab] = np.mean(sub_df['hit'])

            return res

        def eval_sub_data(sub_data):
            lt = len(sub_data)
            for i in range(lt):
                item = sub_data.iloc[i]
                pred = item['prediction']
                gt = item['answer']
                if pred != gt:
                    return 0
            return 1

        def show_result(ret_json):
            show_dict = ret_json.copy()
            table = Table(title=f' GeoChat-Bench ({self.data_file}) ')
            console = Console()
            table.add_column('Category', justify='left')
            table.add_column('Accuracy (%)', justify='right')
            average = show_dict.pop('Average') * 100
            table.add_row('Average', f'{average:.1f}')
            table.add_section()
            for cat_name, cat_acc in show_dict.items():
                table.add_row(cat_name, f'{cat_acc * 100:.1f}')
            with console.capture() as capture:
                console.print(table, end='')
            print('\n' + capture.get())

        data = results_df.sort_values(by='question_id')
        data['prediction'] = [str(x) for x in data['prediction']]

        data_main = data[data['question_id'] < int(1e6)]
        cate_map = {
            i: c
            for i, c in zip(data_main['question_id'], data_main['category'])
        }

        lt = len(data_main)
        hit, tot = 0, 0
        result = {}
        for i in range(lt):
            item = data_main.iloc[i]
            idx = item['question_id']
            assert idx not in result
            sub_data = data_main[data_main['question_id'] % int(1e6) == idx]
            ret = eval_sub_data(sub_data)
            result[idx] = ret
            hit += ret
            tot += 1

        data_main = data_main.copy()
        data_main['hit'] = [result[i] for i in data_main['question_id']]
        data_main['category'] = [cate_map[i] for i in data_main['question_id']]

        ret_json = calc_acc(data_main, 'overall')
        leaf = calc_acc(data_main, 'category')
        ret_json.update(leaf)
        if show:
            show_result(ret_json)
        return ret_json
