# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from glob import glob
import os
import ujson

import argparse
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap

from comp_med_dsum_eval.eval.rouge import preprocess_sentence
from comp_med_dsum_eval.preprocess.entity.process_ents import filter_ents, get_vocab_info
from comp_med_dsum_eval.perturber.process_ents_for_perturbed import (
    flatten_ents, process_ents, compute_scores, get_cache_key
)


def process(data_dir, out_dir, ent_fn, wv, tf_idf_map, default_idf, vocab):
    example_id = ent_fn.split('/')[-1].replace('.json', '')
    meta_out_fn = os.path.join(out_dir, f'{example_id}.csv')

    original_ent_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.json')
    with open(original_ent_fn, 'r') as fd:
        original_ents = ujson.load(fd)

    source_ents = process_ents(original_ents['source'], vocab, 'source')
    target_ents = process_ents(original_ents['target'], vocab, 'target')
    target_ent_objs = [
        {'ents': target_ents['ents'][j], 'toks': target_ents['toks'][j]} for j in range(len(target_ents['ents']))]

    merge_scores = []
    with open(ent_fn, 'r') as fd:
        try:
            sum_ents = ujson.load(fd)
        except:
            print(ent_fn)
            print(example_id)
            raise
    sum_ents_flat = filter_ents(flatten_ents(sum_ents['ents']))
    sum_texts = list(map(lambda x: x['Text'], sum_ents_flat))
    sum_texts_cleaned = list(map(lambda x: preprocess_sentence(x, vocab_filter=vocab), sum_texts))

    sum_toks = [x.split(' ') for x in sum_texts_cleaned]
    cached_scores = defaultdict(dict)

    for i, sum_ent_obj in enumerate(sum_ents_flat):
        tok_p = [x for x in sum_toks[i] if len(x) > 0]
        revised_key = get_cache_key(sum_ent_obj, tok_p)
        for source_idx, source_ent_obj in enumerate(source_ents['ents']):
            tok_s = [x for x in source_ents['toks'][source_idx] if len(x) > 0]
            source_key = get_cache_key(source_ent_obj, tok_s)
            if source_key in cached_scores[revised_key]:
                scores = cached_scores[revised_key][source_key]
            else:
                scores = compute_scores(tok_s, tok_p, source_ent_obj, sum_ent_obj, wv, tf_idf_map, default_idf)
                cached_scores[revised_key][source_key] = scores
            if scores['should_merge']:  # Files become way too large otherwise
                row = {
                    'source_ent_id': source_ent_obj['ent_id'],
                    'sum_ent_id': sum_ent_obj['ent_id'],
                    'source_text': source_ent_obj['Text'],
                    'sum_text': sum_ent_obj['Text'],
                    'relation': 'source-sum'
                }
                row.update(scores)
                merge_scores.append(row)
        for target_ent_obj in target_ent_objs:
            tok_t = [x for x in target_ent_obj['toks'] if len(x) > 0]
            scores = compute_scores(
                tok_t, tok_p, target_ent_obj['ents'], sum_ent_obj, wv, tf_idf_map, default_idf)
            if scores['should_merge']:
                row = {
                    'target_ent_id': target_ent_obj['ents']['ent_id'],
                    'sum_ent_id': sum_ent_obj['ent_id'],
                    'target_text': target_ent_obj['ents']['Text'],
                    'sum_text': sum_ent_obj['Text'],
                    'relation': 'target-sum'
                }
                row.update(scores)
                merge_scores.append(row)
    if len(merge_scores) > 0:
        df = pd.DataFrame(merge_scores)
        df.to_csv(meta_out_fn, index=False)
        return 1
    else:
        print(f'Empty example. Nothing to save --> {example_id}')
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process (merge) entities for model-generated summaries.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--cpu_frac', default=0.75, type=float)

    args = parser.parse_args()
    data_dir = os.path.join(args.input_dir, args.target)
    gen_fn = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment, 'outputs.csv')
    in_dir = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment, 'acm_output')
    out_dir = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment, 'ent_merges')
    os.makedirs(out_dir, exist_ok=True)

    wv, tf_idf_map, default_idf, vocab = get_vocab_info(data_dir)

    pattern = in_dir + '/*.json'
    sum_fns = glob(pattern)
    print(f'Found {len(sum_fns)} extracted files to process for entity extraction...')

    if args.cpu_frac == -1:
        statuses = list(tqdm(map(
            lambda fn: process(data_dir, out_dir, fn, wv, tf_idf_map, default_idf, vocab), sum_fns),
            total=len(sum_fns)))
    else:
        statuses = list(p_uimap(
            lambda fn: process(data_dir, out_dir, fn, wv, tf_idf_map, default_idf, vocab), sum_fns,
            num_cpus=args.cpu_frac))

    print(f'Successfully completed processing {sum(statuses)} examples')
