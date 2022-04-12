# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os
import ujson

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap


from comp_med_dsum_eval.preprocess.entity.extract_ents import get_prod_hera_client, get_ents


def extract(example_id, records, client, out_dir):
    outputs = {}
    for record in records:
        assert record['example_id'] == example_id
        if len(record['prediction']) == 0:
            print(f'Empty sentence: {example_id}')
            continue
        ents = get_ents(record['prediction'], client)
        sent_idx = record['target_sent_idx']
        if sent_idx not in outputs:
            outputs[sent_idx] = {
                'revised': {},
                'context': record['context'],
                'target_sent': record['target_sent'],
            }
        source_extract_code = record['source_extract_code']
        for ent_type in ents:
            for ent_idx in range(len(ents[ent_type])):
                s, e = ents[ent_type][ent_idx]['BeginOffset'], ents[ent_type][ent_idx]['EndOffset']
                ents[ent_type][ent_idx]['ent_id'] = f'{ent_type}-sent-{sent_idx}-revise-{source_extract_code}-{s}-{e}'
        outputs[sent_idx]['revised'][source_extract_code] = {
            'ents': ents,
            'text': record['prediction'],
        }
    out_fn = os.path.join(out_dir, f'{example_id}.json')
    with open(out_fn, 'w') as fd:
        ujson.dump(outputs, fd)
    return 1


def read_fn(fn):
    df = pd.read_csv(fn)
    df['prediction'].fillna('', inplace=True)
    return fn, df, len(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extracting entities for revised sentences.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default='yay')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.33, type=float)
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('--chunksize', default=3, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)

    args = parser.parse_args()
    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)
    revise_dir = os.path.join(data_dir, 'revise', 'output', args.experiment)
    out_dir = os.path.join(data_dir, 'revise', 'acm_output', args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    pattern = revise_dir + '/*.csv'
    revised_fns = glob(pattern)
    print(f'Found {len(revised_fns)} files to process for entity extraction...')

    if args.debug:
        debug_n = 128
        print(f'Debugging with first {debug_n} files...')
        revised_fns = revised_fns[:min(len(revised_fns), debug_n)]

    if args.only_new:
        example_ids = [x.split('/')[-1].replace('.csv', '') for x in revised_fns]
        json_pattern = out_dir + '/*.json'
        ent_fns = glob(json_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.json', '') for x in ent_fns])
        print(f'Choosing not to re extract entities for the {len(done_example_ids)} already completed examples')
        revised_fns = [fn for fn, example_id in zip(revised_fns, example_ids) if example_id not in done_example_ids]
        if len(revised_fns) == 0:
            print('No new files to extract for.  Exiting successfully!')
            exit(0)

    # Generate for a specific chunk_idx of the data, with size specified by chunksize
    if args.chunk_idx is not None:
        example_ids = list(np.sort([x.split('/')[-1].replace('.csv', '') for x in revised_fns]))
        chunks = np.array_split(example_ids, args.chunksize)
        example_id_set = set(chunks[args.chunk_idx])
        revised_fns = [
            revised_fn for revised_fn, example_id in zip(revised_fns, example_ids) if example_id in example_id_set]
        print(f'Filtering for dataset chunk index {args.chunk_idx + 1}/{args.chunksize} of size {len(revised_fns)}')

    client = get_prod_hera_client()

    if args.cpu_frac == -1:
        data_frames = list(sorted(list(tqdm(map(read_fn, revised_fns))), key=lambda x: x[-1]))
        statuses = list(tqdm(map(
            lambda x: extract(x[0].split('/')[-1].replace('.csv', ''), x[1].to_dict('records'), client, out_dir),
            data_frames), total=len(revised_fns)))
    else:
        data_frames = list(sorted(list(p_uimap(read_fn, revised_fns, num_cpus=args.cpu_frac)), key=lambda x: x[-1]))
        statuses = list(p_uimap(
            lambda x: extract(x[0].split('/')[-1].replace('.csv', ''), x[1].to_dict('records'), client, out_dir),
            data_frames, num_cpus=args.cpu_frac))

    print(f'Successfully completed processing {sum(statuses)} examples')
