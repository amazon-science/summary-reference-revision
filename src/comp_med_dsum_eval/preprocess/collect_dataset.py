# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import pandas as pd
from p_tqdm import p_uimap


def add_ids(html_str):
    tps = html_str.split('<SEP>')
    output_tps = []
    curr_sent_idx = -1
    curr_sec_idx = -1
    curr_note_idx = -1
    curr_para_idx = -1
    for tp in tps:
        if tp.startswith('<d'):
            assert ' idx=' not in tp
            curr_note_idx += 1
            output_tps.append(tp.rstrip('>') + f' idx={curr_note_idx}>')
        elif tp.startswith('<h'):
            assert ' idx=' not in tp
            curr_sec_idx += 1
            output_tps.append(tp.rstrip('>') + f' idx={curr_sec_idx}>')
        elif tp.startswith('<p'):
            assert ' idx=' not in tp
            curr_para_idx += 1
            output_tps.append(tp.rstrip('>') + f' idx={curr_para_idx}>')
        elif tp.startswith('<s'):
            assert ' idx=' not in tp
            curr_sent_idx += 1
            output_tps.append(f'<s idx={curr_sent_idx}>')
        else:
            output_tps.append(tp)
    return '<SEP>'.join(output_tps)


def merge_source_target(example_id, source_df, target_df):
    source_notes = source_df.sort_values(by='CHARTDATE')['TEXT'].tolist()
    target = str(target_df['target'].tolist()[0])
    return {
        'example_id': example_id,
        'source': add_ids('<SEP>'.join(source_notes)),
        'target': add_ids(target)
    }


def add_example_id(df):
    df['example_id'] = df['SUBJECT_ID'].combine(df['HADM_ID'], lambda a, b: f'{str(int(a))}_{str(int(b))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Structuring notes by section & sentence.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, args.target)

    source_fn = os.path.join(input_dir, 'source_notes_tagged.csv')
    target_fn = os.path.join(input_dir, 'target_notes_tagged.csv')

    print('Loading source notes...')
    source_df = pd.read_csv(source_fn)
    source_df.dropna(subset=['TEXT'], inplace=True)
    source_df = source_df[source_df['ISERROR'] != 1]
    source_df['CHARTDATE'] = pd.to_datetime(source_df.CHARTDATE)
    print('Loading target notes...')
    target_df = pd.read_csv(target_fn)
    target_df.dropna(subset=['target'], inplace=True)

    print('Generating Example IDs...')
    add_example_id(source_df)
    add_example_id(target_df)

    print('Grouping by Example IDs...')
    example2source = dict(tuple(source_df.groupby('example_id')))
    example2target = dict(tuple(target_df.groupby('example_id')))

    source_ids = set(example2source.keys())
    print(f'{len(source_ids)} source examples')
    target_ids = set(example2target.keys())
    print(f'{len(target_ids)} target examples')
    mutual_ids = list(source_ids.intersection(target_ids))
    print(f'{len(mutual_ids)} shared examples')

    out_fn = os.path.join(input_dir, 'summary_dataset.csv')
    print(f'Merging source & target to {out_fn}')
    summary_df = pd.DataFrame(
        list(p_uimap(lambda example_id: merge_source_target(
            example_id, example2source[example_id], example2target[example_id]), mutual_ids, num_cpus=args.cpu_frac)))
    summary_df.dropna(inplace=True)
    print(f'Saving merged dataframe with {len(summary_df)} examples to {out_fn}')
    summary_df.to_csv(out_fn, index=False)
