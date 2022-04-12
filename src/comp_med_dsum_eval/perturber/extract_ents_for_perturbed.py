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
        if len(record['text']) == 0:
            print(f'Empty sentence: {example_id}')
            continue
        ents = get_ents(record['text'], client)
        sent_idx = record['sent_idx']
        if sent_idx not in outputs:
            outputs[sent_idx] = {
                'original': record['text_original'],
                'perturbed': {}
            }
        perturb_idx = record['perturb_idx']
        for ent_type in ents:
            for ent_idx in range(len(ents[ent_type])):
                s, e = ents[ent_type][ent_idx]['BeginOffset'], ents[ent_type][ent_idx]['EndOffset']
                ents[ent_type][ent_idx]['ent_id'] = f'{ent_type}-sent-{sent_idx}-perturb-{perturb_idx}-{s}-{e}'
        outputs[sent_idx]['perturbed'][record['perturb_idx']] = {
            'ents': ents,
            'text': record['text'],
        }
    out_fn = os.path.join(out_dir, f'{example_id}.json')
    with open(out_fn, 'w') as fd:
        ujson.dump(outputs, fd)
    return 1


def read_fn(fn):
    df = pd.read_csv(fn)
    df['text'].fillna('', inplace=True)
    return fn, df, len(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extracting entities to be perturbed')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default='ent_sample')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('--chunksize', default=3, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)

    args = parser.parse_args()
    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)
    noise_dir = os.path.join(data_dir, 'perturb', args.experiment, 'output')
    out_dir = os.path.join(data_dir, 'perturb', args.experiment, 'acm_output')
    os.makedirs(out_dir, exist_ok=True)

    pattern = noise_dir + '/*.csv'
    noise_fns = glob(pattern)
    print(f'Found {len(noise_fns)} files to process for entity extraction...')

    if args.only_new:
        example_ids = [x.split('/')[-1].replace('.csv', '') for x in noise_fns]
        json_pattern = out_dir + '/*.json'
        ent_fns = glob(json_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.json', '') for x in ent_fns])
        print(f'Choosing not to re extract entities for the {len(done_example_ids)} already completed examples')
        noise_fns = [noise_fn for noise_fn, example_id in zip(noise_fns, example_ids)
                     if example_id not in done_example_ids]
        if len(noise_fns) == 0:
            print('No new files to extract for.  Exiting successfully!')
            exit(0)

    # Generate for a specific chunk_idx of the data, with size specified by chunksize
    if args.chunk_idx is not None:
        example_ids = list(np.sort([x.split('/')[-1].replace('.csv', '') for x in noise_fns]))
        chunks = np.array_split(example_ids, args.chunksize)
        example_id_set = set(chunks[args.chunk_idx])
        noise_fns = [noise_fn for noise_fn, example_id in zip(noise_fns, example_ids)
                     if example_id in example_id_set]
        print(f'Filtering for dataset chunk index {args.chunk_idx + 1}/{args.chunksize} of size {len(noise_fns)}')

    client = get_prod_hera_client()

    if args.cpu_frac == -1:
        data_frames = list(sorted(list(tqdm(map(read_fn, noise_fns))), key=lambda x: x[-1]))
        statuses = list(tqdm(map(
            lambda x: extract(x[0].split('/')[-1].replace('.csv', ''), x[1].to_dict('records'), client, out_dir),
            data_frames), total=len(noise_fns)))
    else:
        data_frames = list(sorted(list(p_uimap(read_fn, noise_fns, num_cpus=args.cpu_frac)), key=lambda x: x[-1]))
        statuses = list(p_uimap(
            lambda x: extract(x[0].split('/')[-1].replace('.csv', ''), x[1].to_dict('records'), client, out_dir),
            data_frames, num_cpus=args.cpu_frac))

    print(f'Successfully completed processing {sum(statuses)} examples')
