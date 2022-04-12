# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ujson
from glob import glob

import argparse
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm

from comp_med_dsum_eval.preprocess.entity.extract_ents import get_prod_hera_client, extract_ents


def process(record, out_dir):
    text = record['TEXT']
    client = get_prod_hera_client()
    ents = extract_ents(text, client)
    note_id = record['note_id']
    out_fn = os.path.join(out_dir, f'{note_id}.json')
    with open(out_fn, 'w') as fd:
        ujson.dump(ents, fd)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extracting entities with Comprehend Medical from MIMIC')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--cpu_frac', default=1.0, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, 'dsum')
    mini_str = '_mini' if args.debug else '_filt'
    in_fn = os.path.join(input_dir, f'dsum_tagged{mini_str}.csv')
    out_dir = os.path.join(input_dir, 'acm_output')

    processed_fns = glob(os.path.join(out_dir, '*.json'))
    processed_note_ids = set([x.split('/')[-1].replace('.json', '') for x in processed_fns])

    os.makedirs(out_dir, exist_ok=True)
    print('Reading in {}'.format(in_fn))
    df = pd.read_csv(in_fn)
    prev_n = len(df)
    df.dropna(subset=['TEXT'], inplace=True)
    df = df.sort_values(by='TEXT', key=lambda x: x.str.len())
    n = len(df)
    print(f'{prev_n - n} rows with empty source or target')
    df = df[~df['note_id'].isin(processed_note_ids)]
    print(f'Processing {len(df)} records instead of {n} because some have already been processed')
    records = df.to_dict('records')
    print('Processing {} notes'.format(len(df)))
    if args.cpu_frac == -1:
        num_ents = list(tqdm(map(lambda record: process(record, out_dir), records)))
    else:
        num_ents = list(
            p_uimap(lambda record: process(record, out_dir), records, num_cpus=args.cpu_frac))

    print(sum(num_ents))
