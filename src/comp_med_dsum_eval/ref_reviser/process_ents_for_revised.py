# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from glob import glob
import os
import ujson
import regex as re

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
    merge_scores = []
    with open(ent_fn, 'r') as fd:
        try:
            example_ents = ujson.load(fd)
        except:
            print(ent_fn)
            print(example_id)
            raise
    cached_scores = defaultdict(dict)
    for sent_idx in example_ents:
        revised_objs = example_ents[sent_idx]['revised']
        original_obj = example_ents[sent_idx]['target_sent']
        rel_target_ents = [
            {'ents': target_ents['ents'][j], 'toks': target_ents['toks'][j]} for j in range(len(target_ents['ents']))
            if target_ents['ents'][j]['sent_idx'] == int(sent_idx)]
        assert len(re.findall(r'</e>', original_obj)) == len(rel_target_ents)
        for revise_idx, revised_ents in revised_objs.items():
            revised_ents_flat = filter_ents(flatten_ents(revised_ents['ents']))
            revised_texts = list(map(lambda x: x['Text'], revised_ents_flat))
            revised_texts_cleaned = list(map(lambda x: preprocess_sentence(x, vocab_filter=vocab), revised_texts))

            revised_toks = [x.split(' ') for x in revised_texts_cleaned]
            for i, revised_ent_obj in enumerate(revised_ents_flat):
                tok_p = [x for x in revised_toks[i] if len(x) > 0]
                revised_key = get_cache_key(revised_ent_obj, tok_p)
                for source_idx, source_ent_obj in enumerate(source_ents['ents']):
                    tok_s = [x for x in source_ents['toks'][source_idx] if len(x) > 0]
                    source_key = get_cache_key(source_ent_obj, tok_s)
                    if source_key in cached_scores[revised_key]:
                        scores = cached_scores[revised_key][source_key]
                    else:
                        scores = compute_scores(tok_s, tok_p, source_ent_obj, revised_ent_obj, wv, tf_idf_map, default_idf)
                        cached_scores[revised_key][source_key] = scores
                    if scores['should_merge']:  # Files become way too large otherwise
                        row = {
                            'source_ent_id': source_ent_obj['ent_id'],
                            'revised_ent_id': revised_ent_obj['ent_id'],
                            'source_text': source_ent_obj['Text'],
                            'revised_text': revised_ent_obj['Text'],
                            'relation': 'source-revised'
                        }
                        row.update(scores)
                        merge_scores.append(row)
                for target_ent_obj in rel_target_ents:
                    tok_t = [x for x in target_ent_obj['toks'] if len(x) > 0]
                    scores = compute_scores(
                        tok_t, tok_p, target_ent_obj['ents'], revised_ent_obj, wv, tf_idf_map, default_idf)
                    if scores['should_merge']:
                        row = {
                            'target_ent_id': target_ent_obj['ents']['ent_id'],
                            'revised_ent_id': revised_ent_obj['ent_id'],
                            'target_text': target_ent_obj['ents']['Text'],
                            'revised_text': revised_ent_obj['Text'],
                            'relation': 'target-revised'
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
    parser = argparse.ArgumentParser('Processing Entities (i.e., compute merge scores) for Perturbed Sentences')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default='yay')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-only_new', default=False, action='store_true')

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)
    in_dir = os.path.join(data_dir, 'revise', 'acm_output', args.experiment)
    out_dir = os.path.join(data_dir, 'revise', 'ent_merges', args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    wv, tf_idf_map, default_idf, vocab = get_vocab_info(data_dir)

    pattern = in_dir + '/*.json'
    revise_fns = glob(pattern)
    print(f'Found {len(revise_fns)} extracted files to process for entity extraction...')

    if args.only_new:
        example_ids = [x.split('/')[-1].replace('.json', '') for x in revise_fns]
        meta_pattern = out_dir + '/*.csv'
        meta_ent_fns = glob(meta_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in meta_ent_fns])
        print(f'Choosing not to re extract entities for the {len(done_example_ids)} already completed examples')
        revise_fns = [fn for fn, example_id in zip(revise_fns, example_ids) if example_id not in done_example_ids]

    if args.debug:
        args.cpu_frac = -1
        revise_fns = revise_fns[:min(10, len(revise_fns))]

    if args.cpu_frac == -1:
        statuses = list(tqdm(map(
            lambda fn: process(data_dir, out_dir, fn, wv, tf_idf_map, default_idf, vocab), revise_fns),
            total=len(revise_fns)))
    else:
        statuses = list(p_uimap(
            lambda fn: process(data_dir, out_dir, fn, wv, tf_idf_map, default_idf, vocab), revise_fns,
            num_cpus=args.cpu_frac))

    print(f'Successfully completed processing {sum(statuses)} examples')
