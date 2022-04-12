# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse

import numpy as np
import pandas as pd
np.random.seed(1992)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate dataset splits...')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--splits', default='0.95,0.025,0.025')

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)

    in_fn = os.path.join(data_dir, 'summary_dataset.csv')
    print(f'Loading in data from {in_fn}')
    data_df = pd.read_csv(in_fn)
    data_df = data_df.assign(
        patient_id=data_df['example_id'].apply(lambda x: x.split('_')[0]),
        visit_id=data_df['example_id'].apply(lambda x: x.split('_')[1]),
    )
    print(f'Loaded {len(data_df)} examples')
    viable_visits = set(pd.read_csv(os.path.join(args.input_dir, 'viable_visits.csv'))['HADM_ID'].apply(
        lambda x: str(x).split('.')[0]))
    data_df = data_df.assign(is_viable=data_df['visit_id'].apply(lambda x: x in viable_visits))
    splits = ['train', 'validation', 'test']
    split_fracs = [float(x) for x in args.splits.split(',')]
    assert sum(split_fracs) == 1.0
    available_patient_ids = set(data_df['patient_id'].unique())
    n = len(available_patient_ids)
    for split, frac in zip(splits, split_fracs):
        target_num = round(frac * n)
        curr_n = len(available_patient_ids)
        split_patient_ids = available_patient_ids
        if target_num < curr_n:
            split_patient_ids = set(np.random.choice(list(available_patient_ids), size=(target_num,), replace=False))
        split_df = data_df[data_df['patient_id'].isin(split_patient_ids)]
        meta_out_fn = os.path.join(data_dir, f'{split}_example_ids.csv')
        out_fn = os.path.join(data_dir, f'summary_dataset_{split}.csv')
        print(f'Saving {len(split_df)} {split} examples to {meta_out_fn} and {out_fn}')
        split_df[['example_id', 'patient_id', 'visit_id']].to_csv(meta_out_fn, index=False)
        available_patient_ids -= split_patient_ids
        split_df.to_csv(out_fn, index=False)
