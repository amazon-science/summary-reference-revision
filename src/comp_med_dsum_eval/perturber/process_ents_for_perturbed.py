# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import itertools
from glob import glob
import os
import ujson
import regex as re

import argparse
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap

from comp_med_dsum_eval.eval.rouge import preprocess_sentence
from comp_med_dsum_eval.preprocess.entity.process_ents import (
    add_ent_id, compute_code_overlap, is_match, filter_ents, get_vocab_info
)


def flatten_ents(ent_obj):
    flat = []
    for k, v_arr in ent_obj.items():
        assert k != 'entity'
        for v in v_arr:
            v['dtype'] = k
            flat.append(v)
    return flat


def process_ents(ents, vocab, version='source'):
    ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, version)) for ent_obj in ents]
    ))
    texts = list(map(lambda x: x['Text'], ents_flat))
    texts_cleaned = list(map(lambda x: preprocess_sentence(x, vocab_filter=vocab), texts))
    toks = [x.split(' ') for x in texts_cleaned]
    return {'ents': ents_flat, 'texts': texts, 'toks': toks}


def compute_scores(tok_s, tok_t, source_ent, target_ent, wv, tf_idf_map, default_idf):
    target_denom = sum([tf_idf_map.get(tok, default_idf) for tok in tok_t])
    source_denom = sum([tf_idf_map.get(tok, default_idf) for tok in tok_s])
    overlaps = list(set(tok_t).intersection(tok_s))
    overlap_idf = sum([tf_idf_map.get(tok, default_idf) for tok in overlaps])
    zero_denom = (target_denom + source_denom) == 0
    tok_overlap = 0 if zero_denom else overlap_idf / (0.5 * source_denom + 0.5 * target_denom)
    lex_sim = 0 if min(len(tok_s), len(tok_t)) == 0 else wv.n_similarity(tok_s, tok_t)

    code_overlap = compute_code_overlap(target_ent, source_ent)
    if code_overlap is None:
        agg_score = (tok_overlap + lex_sim) / 2.0
    else:
        agg_score = (code_overlap + tok_overlap + lex_sim) / 3.0

    should_merge = is_match(lex_sim, tok_overlap, code_overlap, agg_score)
    return {
        'code_overlap': code_overlap,
        'tok_overlap': tok_overlap,
        'lex_sim': lex_sim,
        'agg_score': agg_score,
        'should_merge': should_merge,
    }


def get_cache_key(ent_obj, toks):
    if len(toks) == 0:
        return ent_obj['dtype'] + '_' + ent_obj['Text']
    return ent_obj['dtype'] + '_' + '_'.join(toks)


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
        perturbed_objs = example_ents[sent_idx]['perturbed']
        original_obj = example_ents[sent_idx]['original']
        rel_target_ents = [
            {'ents': target_ents['ents'][j], 'toks': target_ents['toks'][j]} for j in range(len(target_ents['ents']))
            if target_ents['ents'][j]['sent_idx'] == int(sent_idx)]
        assert len(re.findall(r'</e>', original_obj)) == len(rel_target_ents)
        for perturb_idx, perturbed_ents in perturbed_objs.items():
            perturbed_ents_flat = filter_ents(flatten_ents(perturbed_ents['ents']))
            perturbed_texts = list(map(lambda x: x['Text'], perturbed_ents_flat))
            perturbed_texts_cleaned = list(map(lambda x: preprocess_sentence(x, vocab_filter=vocab), perturbed_texts))

            perturbed_toks = [x.split(' ') for x in perturbed_texts_cleaned]
            for i, perturb_ent_obj in enumerate(perturbed_ents_flat):
                tok_p = [x for x in perturbed_toks[i] if len(x) > 0]
                perturb_key = get_cache_key(perturb_ent_obj, tok_p)
                for source_idx, source_ent_obj in enumerate(source_ents['ents']):
                    tok_s = [x for x in source_ents['toks'][source_idx] if len(x) > 0]
                    source_key = get_cache_key(source_ent_obj, tok_s)
                    if source_key in cached_scores[perturb_key]:
                        scores = cached_scores[perturb_key][source_key]
                    else:
                        scores = compute_scores(tok_s, tok_p, source_ent_obj, perturb_ent_obj, wv, tf_idf_map, default_idf)
                        cached_scores[perturb_key][source_key] = scores
                    if scores['should_merge']:  # Files become way too large otherwise
                        row = {
                            'source_ent_id': source_ent_obj['ent_id'],
                            'perturbed_ent_id': perturb_ent_obj['ent_id'],
                            'source_text': source_ent_obj['Text'],
                            'perturbed_text': perturb_ent_obj['Text'],
                            'relation': 'source-perturbed'
                        }
                        row.update(scores)
                        merge_scores.append(row)
                for target_ent_obj in rel_target_ents:
                    tok_t = [x for x in target_ent_obj['toks'] if len(x) > 0]
                    scores = compute_scores(
                        tok_t, tok_p, target_ent_obj['ents'], perturb_ent_obj, wv, tf_idf_map, default_idf)
                    if scores['should_merge']:
                        row = {
                            'target_ent_id': target_ent_obj['ents']['ent_id'],
                            'perturbed_ent_id': perturb_ent_obj['ent_id'],
                            'target_text': target_ent_obj['ents']['Text'],
                            'perturbed_text': perturb_ent_obj['Text'],
                            'relation': 'target-perturbed'
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
    parser.add_argument('--experiment', default='ent_sample')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-only_new', default=False, action='store_true')

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)
    in_dir = os.path.join(data_dir, 'perturb', args.experiment, 'acm_output')
    out_dir = os.path.join(data_dir, 'perturb', args.experiment, 'ent_merges')
    os.makedirs(out_dir, exist_ok=True)

    wv, tf_idf_map, default_idf, vocab = get_vocab_info(data_dir)

    pattern = in_dir + '/*.json'
    noise_fns = glob(pattern)
    print(f'Found {len(noise_fns)} extracted files to process for entity extraction...')

    if args.only_new:
        example_ids = [x.split('/')[-1].replace('.json', '') for x in noise_fns]
        meta_pattern = out_dir + '/*.csv'
        meta_ent_fns = glob(meta_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in meta_ent_fns])
        print(f'Choosing not to re extract entities for the {len(done_example_ids)} already completed examples')
        noise_fns = [noise_fn for noise_fn, example_id in zip(noise_fns, example_ids)
                     if example_id not in done_example_ids]

    if args.debug:
        args.cpu_frac = -1
        noise_fns = noise_fns[:min(10, len(noise_fns))]

    if args.cpu_frac == -1:
        statuses = list(tqdm(map(
            lambda fn: process(data_dir, out_dir, fn, wv, tf_idf_map, default_idf, vocab), noise_fns),
            total=len(noise_fns)))
    else:
        statuses = list(p_uimap(
            lambda fn: process(data_dir, out_dir, fn, wv, tf_idf_map, default_idf, vocab), noise_fns,
            num_cpus=args.cpu_frac))

    print(f'Successfully completed processing {sum(statuses)} examples')
