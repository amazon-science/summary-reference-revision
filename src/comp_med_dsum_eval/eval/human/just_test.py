# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

from tqdm import tqdm
from glob import glob
import pandas as pd

DATA_DIR = '/efs/griadams/bhc'


def add_example_id(df):
    df['example_id'] = df['SUBJECT_ID'].combine(df['HADM_ID'], lambda a, b: f'{str(int(a))}_{str(int(b))}')


if __name__ == '__main__':
    test_fn = os.path.join(DATA_DIR, 'mimic_sum', 'results', 'long_revised_balanced', 'outputs.csv')
    test_example_ids = pd.read_csv(test_fn)['example_id'].tolist()
    in_dir = os.path.join(DATA_DIR, 'embed_cache')
    out_dir = os.path.join(DATA_DIR, 'embed_cache_test')

    os.makedirs(out_dir, exist_ok=True)
    for example_id in tqdm(test_example_ids):
        from_fn = os.path.join(in_dir, f'{example_id}.json')
        to_fn = os.path.join(out_dir, f'{example_id}.json')
        assert os.path.exists(from_fn)
        shutil.copy(from_fn, to_fn)
    assert len(glob(os.path.join(out_dir, '*.json'))) == len(test_example_ids)
