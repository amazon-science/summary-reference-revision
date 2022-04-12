# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os
import random

import numpy as np
import pandas as pd
import torch


def filter_df_for_eval(data_df, data_dir):
    eval_uids = set(pd.read_csv(os.path.join(data_dir, 'high_quality', 'eval_examples.csv'))['uid'])
    print(f'Choosing to generate for only the {len(eval_uids)} eval sentences')
    data_df = data_df[data_df['sent_uid'].isin(eval_uids)]
    print(f'{len(data_df)} sentences remaining after filtering for ones in evaluation set')
    return data_df


def filter_out_done(data_df, done_dir, suffix='csv'):
    csv_pattern = done_dir + f'/*.{suffix}'
    ent_fns = glob(csv_pattern)
    done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in ent_fns])
    print(f'Choosing not to re-generate perturbations for the {len(done_example_ids)} already completed examples')
    prev_n = len(data_df)
    data_df = data_df[~data_df['example_id'].isin(done_example_ids)]
    n = len(data_df)
    print(f'Shrunk dataframe from {prev_n} to {n}')
    return data_df


def get_chunk(data_df, chunk_idx, chunksize, sort_first=True):
    if sort_first:
        example_ids = list(np.sort(data_df['example_id'].unique().tolist()))
    else:
        example_ids = list(data_df['example_id'].unique().tolist())
    chunks = np.array_split(example_ids, chunksize)
    example_id_set = set(chunks[chunk_idx])
    data_df = data_df[data_df['example_id'].isin(example_id_set)]
    print(f'Filtering for dataset chunk index {chunk_idx + 1}/{chunksize} of size {len(data_df)}')
    return data_df


def set_seeds(seed=1992):
    # Set random seed across generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
