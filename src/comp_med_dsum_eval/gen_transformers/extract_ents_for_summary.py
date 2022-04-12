# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ujson

import argparse
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap


from comp_med_dsum_eval.preprocess.entity.extract_ents import get_prod_hera_client, get_ents


def extract(record, client, out_dir):
    example_id = record['example_id']
    ents = get_ents(record['prediction'], client)
    for ent_type in ents:
        for ent_idx in range(len(ents[ent_type])):
            s, e = ents[ent_type][ent_idx]['BeginOffset'], ents[ent_type][ent_idx]['EndOffset']
            ents[ent_type][ent_idx]['ent_id'] = f'{ent_type}-{s}-{e}'
    outputs = {
        'ents': ents,
        'text': record['prediction'],
    }
    out_fn = os.path.join(out_dir, f'{example_id}.json')
    with open(out_fn, 'w') as fd:
        ujson.dump(outputs, fd)
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract entities for model-generated summaries.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--cpu_frac', default=0.75, type=float)

    args = parser.parse_args()
    data_dir = os.path.join(args.input_dir, args.target)
    gen_fn = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment, 'outputs.csv')
    out_dir = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment, 'acm_output')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(gen_fn)
    records = df[['example_id', 'prediction']].to_dict('records')

    client = get_prod_hera_client()

    if args.cpu_frac == -1:
        statuses = list(tqdm(map(
            lambda x: extract(x[0].split('/')[-1].replace('.csv', ''), x[1].to_dict('records'), client, out_dir),
            records), total=len(records)))
    else:
        statuses = list(p_uimap(
            lambda record: extract(record, client, out_dir), records, num_cpus=args.cpu_frac))

    print(f'Successfully completed processing {sum(statuses)} examples')
