# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import ujson
import os

import argparse
import numpy as np
np.random.seed(1992)
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm

from comp_med_dsum_eval.preprocess.entity.process_ents import add_ent_id, filter_ents
from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments
from comp_med_dsum_eval.ref_reviser.dataset import tokenize, remove_tags_from_sent
from comp_med_dsum_eval.gen_transformers.dataset import add_meta


def process(record, data_dir):
    revise_dir = os.path.join(data_dir, 'revise', 'contexts')
    example_id = record['example_id']
    fn = os.path.join(revise_dir, f'{example_id}.json')
    with open(fn, 'r') as fd:
        examples = ujson.load(fd)
    if len(examples) == 0:
        return {}
    example_id = examples[0]['example_id']
    ent_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.json')
    with open(ent_fn) as fd:
        ents = ujson.load(fd)
    source_ents = ents['source']
    source_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'source')) for ent_obj in source_ents]
    ))
    all_source_ent_ids = [x['ent_id'] for x in source_ents_flat]

    source_toks = tokenize(remove_tags_from_sent(''.join(record['source'].split('<SEP>'))))
    target_toks = tokenize(remove_tags_from_sent(''.join(record['target'].split('<SEP>'))))
    frags = parse_extractive_fragments(source_toks, target_toks, remove_stop=True)
    target_ents = ents['target']
    target_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'target')) for ent_obj in target_ents]
    ))
    all_target_ent_ids = [x['ent_id'] for x in target_ents_flat]
    try:
        orig_merge_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.csv')
        orig_merges = pd.read_csv(orig_merge_fn)
        orig_merges = orig_merges[orig_merges['should_merge']].dropna(subset=['source_ent_id', 'target_ent_id'])
        valid_target_ents = orig_merges['target_ent_id'].unique().tolist()
        covered_source_ents = orig_merges['source_ent_id'].unique().tolist()
    except:
        valid_target_ents = []
        covered_source_ents = []

    num_source_ents = len(all_source_ent_ids)
    num_target_ents = len(all_target_ent_ids)
    halluc_frac = (num_target_ents - len(valid_target_ents)) / max(1, num_target_ents)
    source_ent_cov = len(covered_source_ents) / max(1, num_source_ents)
    stats = [x['stats'] for x in examples]
    stats_df = pd.DataFrame(stats)
    row = {
        'avg_bert_bs_con_cov': stats_df['source_to_target_coverage'].dropna().mean(),
        'target_toks': len(target_toks),
        'num_ents': num_target_ents,
        'source_ent_cov': source_ent_cov,
        'halluc_frac': halluc_frac,
        'ent_rel_frac': 1,  # this is the original reference so it covers all faithful reference entities
    }
    row.update(frags)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Statistics for non-revised references for paper table.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=1.0, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--filter', default=None, choices=['quality', 'admission'])
    parser.add_argument('--max_n', default=None, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)
    in_fn = os.path.join(data_dir, f'summary_dataset{mini_str}.csv')
    print(f'Loading summary dataset from {in_fn}')
    summary_df = pd.read_csv(in_fn)
    summary_df = add_meta(summary_df, data_dir)
    summary_df = summary_df[summary_df['split'] == 'train']
    if args.filter is not None and args.filter == 'admission':
        summary_df = summary_df[summary_df['has_admission_note']]
    elif args.filter is not None and args.filter == 'quality':
        summary_df = summary_df[summary_df['quality'] == 'high']
    if args.max_n is not None and args.max_n < len(summary_df):
        print(f'Randomly shrinking dataset to {args.max_n}')
        summary_df = summary_df.sample(n=args.max_n, replace=False, random_state=1992)
    summary_df = summary_df.sort_values(by='source', key=lambda x: x.str.len())
    records = summary_df.to_dict('records')
    if args.cpu_frac == -1:
        outputs = list(tqdm(map(lambda record: process(record, data_dir), records), total=len(records)))
    else:
        outputs = list(p_uimap(lambda record: process(record, data_dir), records, num_cpus=args.cpu_frac))

    out_df = pd.DataFrame(outputs)
    print('Number of tokens per reference: ', out_df['target_toks'].dropna().mean())
    print('Num entities: ', out_df['num_ents'].mean())
    print('Average extractive coverage: ', out_df['coverage'].mean())
    print('Average extractive density: ', out_df['density'].mean())
    print('Hallucination fraction: ', out_df['halluc_frac'].mean())
    print('Average BertScore coverage: ', out_df['avg_bert_bs_con_cov'].mean())
    print('Source Entity Coverage: ', out_df['source_ent_cov'].mean())
    print('Faithful adjusted recall: ', 1)

    max_n_str = '' if args.max_n is None else '_' + str(args.max_n)
    filter_str = '_original' if args.filter is None else '_' + str(args.filter)
    stats_fn = os.path.join(data_dir, f'stats_for_summary_dataset{filter_str}{mini_str}{max_n_str}.csv')
    print(f'Saving {len(out_df)} reference statistics to {stats_fn}')
    out_df.to_csv(stats_fn, index=False)
