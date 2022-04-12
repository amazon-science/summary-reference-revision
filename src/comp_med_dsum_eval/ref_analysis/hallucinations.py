# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from glob import glob
import os
import ujson

import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm

from comp_med_dsum_eval.preprocess.entity.process_ents import add_ent_id, filter_ents


def process(fn, ent_dir):
    with open(fn, 'r') as fd:
        ents = ujson.load(fd)
    example_id = fn.split('/')[-1].split('.')[0]
    target_ents = ents['target']

    target_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'target')) for ent_obj in target_ents]
    ))

    try:
        merge_scores = pd.read_csv(os.path.join(ent_dir, f'{example_id}.csv'))
        merge_pairs = merge_scores[merge_scores['should_merge']][['source_ent_id', 'target_ent_id']]
        target_overlapping_ent_ids = set(merge_pairs['target_ent_id'].unique())
    except pd.errors.EmptyDataError:
        source_overlapping_ent_ids, target_overlapping_ent_ids, overlapping_target_ents = {}, {}, []

    local_out = []
    global_halluc = 0
    global_n = 0
    for ent_obj in target_ents:
        if ent_obj['dtype'] != 'sent':
            continue
        ents = filter_ents(ent_obj['ents'])
        local_halluc = 0
        local_n = len(ents)
        global_n += len(ents)
        for ent in ents:
            is_hallucinated = 0 if ent['ent_id'] in target_overlapping_ent_ids else 1
            if is_hallucinated:
                local_halluc += 1
                global_halluc += 1

        local_out.append({
            'num_ent': local_n,
            'num_halluc': local_halluc,
            'halluc_frac': local_halluc / max(1, local_n)
        })

    global_out = {
        'num_ent': global_n,
        'num_halluc': global_halluc,
        'halluc_frac': global_halluc / max(1, global_n)
    }
    return global_out, local_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing reference and sentence level hallucination information.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=1.0, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.cpu_frac = -1

    input_dir = os.path.join(args.input_dir, args.target)
    mini_str = '_mini' if args.debug else ''
    ent_dir = os.path.join(input_dir, 'acm_output')
    pattern = ent_dir + '/*.json'
    print(f'Searching for entities in {pattern}...')
    fns = glob(pattern)
    max_n = 25 if args.debug else 100000
    if max_n is not None and max_n < len(fns):
        fns = list(np.random.choice(fns, size=(max_n, ), replace=False))
    if args.cpu_frac == -1:
        outputs = list(tqdm(map(lambda fn: process(fn, ent_dir), fns), total=len(fns)))
    else:
        outputs = list(p_uimap(lambda fn: process(fn, ent_dir), fns, num_cpus=args.cpu_frac))

    global_df = pd.DataFrame([x[0] for x in outputs])
    global_no_ent = len(global_df[global_df['num_ent'] == 0])
    global_df.to_csv('global_halluc.csv', index=False)
    nonzero_global_df = global_df[global_df['num_ent'] > 0]
    print(f'{global_no_ent} references with no entities ({global_no_ent / len(global_df)})')

    global_series = pd.Series(nonzero_global_df['halluc_frac'], name='Reference-Level Hallucination Rates')
    plt.xlim(0, 1)
    plt.ylabel('Reference Counts')
    sns.histplot(global_series, bins=10)
    plt.savefig('global_halluc_rate.png')
    plt.clf()

    local_df = pd.DataFrame(list(itertools.chain(*[x[1] for x in outputs])))
    local_no_ent = len(local_df[local_df['num_ent'] == 0])
    local_one_ent_df = local_df[local_df['num_ent'] == 1]
    local_one_ent = len(local_one_ent_df)
    local_df.to_csv('local_halluc.csv', index=False)
    print(f'{local_no_ent} reference SENTENCES with no entities ({local_no_ent / len(local_df)})')
    nonzero_local_df = local_df[local_df['numlc_ent'] > 0]
    print(f'{local_one_ent} reference SENTENCES with one entity ({local_one_ent / len(local_df)})')
    print('Hallucination rate of one entity sentences: ', local_one_ent_df['halluc_frac'].mean())
    local_series = pd.Series(nonzero_local_df['halluc_frac'], name='Sentence-Level Hallucination Rates')
    plt.xlim(0, 1)
    plt.ylabel('Reference Sentence Counts')
    sns.histplot(local_series, bins=10)
    plt.savefig('local_halluc_rate.png', bbox_inches='tight')
    plt.clf()

    multi_local_df = local_df[local_df['num_ent'] > 1]
    local_multi_series = pd.Series(multi_local_df['halluc_frac'], name='Sentence-Level Hallucination Rates')
    plt.xlim(0, 1)
    plt.ylabel('Reference Sentence Counts')
    sns.histplot(local_multi_series, bins=10)
    plt.savefig('local_halluc_rate_multi_entity.png', bbox_inches='tight')
    plt.clf()
    print('Done!')
