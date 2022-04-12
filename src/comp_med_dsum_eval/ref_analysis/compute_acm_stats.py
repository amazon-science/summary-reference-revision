# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter, defaultdict
import itertools
import os
import ujson
import regex as re
from string import punctuation

import argparse
from lexrank import STOPWORDS
import numpy as np
from nltk import word_tokenize
import pandas as pd
from p_tqdm import p_uimap
from scipy import spatial
from tqdm import tqdm
from scipy.stats import entropy
from scipy.special import kl_div

from comp_med_dsum_eval.eval.rouge import calc_rouge
from comp_med_dsum_eval.preprocess.constants import ACM_ENT_TYPES
from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments
from comp_med_dsum_eval.preprocess.entity.process_ents import add_ent_id, filter_ents
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent, sents_from_html
from comp_med_dsum_eval.gen_transformers.dataset import add_meta


def calculate_redundancy(sents, max_sents=10):
    if len(sents) <= 1:
        return None
    sent_trunc = sents[:min(len(sents), max_sents)]
    sent_pairs = list(itertools.combinations(sent_trunc, 2))
    sent_pairs = sent_pairs[:min(len(sent_pairs), max_sents)]
    n = len(sent_pairs)
    rl = sum([calc_rouge(s1, s2, rouge_types=['rougeL'])['rougeL'].fmeasure for s1, s2 in sent_pairs])
    return rl / n


def get_ent_type_counts(ents):
    counts = {}
    smoothed_counts = {}
    resolved_types = [t for t in ACM_ENT_TYPES if t not in {'GENERIC_NAME', 'BRAND_NAME'}] + ['MEDICATION']
    smooth_constant = 0.1
    smooth_denom = smooth_constant * len(resolved_types)
    for type in resolved_types:
        counts[type] = 0
        smoothed_counts[type] = 0.1
    for ent in ents:
        t = 'MEDICATION' if ent['Type'] in {'GENERIC_NAME', 'BRAND_NAME'} else ent['Type']
        counts[t] += 1
        smoothed_counts[t] += 1

    total_count = len(ents)
    resolved_counts = [counts[t] for t in resolved_types]
    resolved_p = [counts[t] / max(1, total_count) for t in resolved_types]
    smoothed_p = [smoothed_counts[t] / (total_count + smooth_denom) for t in resolved_types]

    return {
        'counts': resolved_counts,
        'p': resolved_p,
        'smooth_p': smoothed_p,
        'types': resolved_types
    }


def compute_corpus_statistics(record):
    example_id, html_source, html_target = record['example_id'], record['source'], record['target']
    stats = {'example_id': example_id}
    ent_dir = os.path.join(data_dir, 'acm_output')

    with open(os.path.join(ent_dir, f'{example_id}.json'), 'r') as fd:
        ents = ujson.load(fd)

    source_ents = ents['source']
    target_ents = ents['target']

    source_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'source')) for ent_obj in source_ents]
    ))
    source_n = len(source_ents_flat)

    target_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'target')) for ent_obj in target_ents]
    ))
    target_n = len(target_ents_flat)

    try:
        merge_scores = pd.read_csv(os.path.join(ent_dir, f'{example_id}.csv'))
        merge_pairs = merge_scores[merge_scores['should_merge']][['source_ent_id', 'target_ent_id']]
        source_overlapping_ent_ids = set(merge_pairs['source_ent_id'].unique())
        target_overlapping_ent_ids = set(merge_pairs['target_ent_id'].unique())
        merged_targets = len(merge_pairs['target_ent_id'].unique())
        overlapping_target_ents = list(filter(lambda x: x['ent_id'] in target_overlapping_ent_ids, target_ents_flat))
    except pd.errors.EmptyDataError:
        assert len(target_ents_flat) == 0 or len(source_ents_flat) == 0
        source_overlapping_ent_ids, target_overlapping_ent_ids, overlapping_target_ents = {}, {}, []
        merged_targets = 0

    source_ent_type_counts = get_ent_type_counts(source_ents_flat)
    target_ent_type_counts = get_ent_type_counts(target_ents_flat)
    overlap_ent_type_counts = get_ent_type_counts(overlapping_target_ents)
    stats['overlap_num_ents'] = len(overlapping_target_ents)

    halluc_rate = 1 if target_n == 0 else (target_n - merged_targets) / target_n
    source_coverage = 1 if source_n == 0 else merged_targets / source_n

    stats['halluc_rate'] = halluc_rate
    stats['source_coverage'] = source_coverage
    stats['ent_compression'] = len(source_ents_flat) / max(1, len(target_ents_flat))
    stats['source_num_ents'] = source_n
    stats['target_num_ents'] = target_n

    ent_type_divergence = sum(kl_div(source_ent_type_counts['smooth_p'], target_ent_type_counts['smooth_p']))
    stats['ent_type_divergence'] = ent_type_divergence

    stats['source_type_entropy'] = entropy(source_ent_type_counts['smooth_p'])
    stats['target_type_entropy'] = entropy(target_ent_type_counts['smooth_p'])
    stats['overlap_type_entropy'] = entropy(overlap_ent_type_counts['smooth_p'])

    for c, p, type in zip(
            source_ent_type_counts['counts'], source_ent_type_counts['p'], source_ent_type_counts['types']):
        stats[f'source_{type}_count'] = c
        stats[f'source_{type}_frac'] = p

    for c, p, type in zip(
            target_ent_type_counts['counts'], target_ent_type_counts['p'], target_ent_type_counts['types']):
        stats[f'target_{type}_count'] = c
        stats[f'target_{type}_frac'] = p

    for c, p, type in zip(
            overlap_ent_type_counts['counts'], overlap_ent_type_counts['p'], overlap_ent_type_counts['types']):
        stats[f'overlap_{type}_count'] = c
        stats[f'overlap_{type}_frac'] = p

    note_types = re.findall(r'note_type=([^ ]+)', record['source'])
    note_types_cts = Counter(note_types)

    source_sents = [remove_tags_from_sent(x) for x in sents_from_html(html_source)]
    target_sents = [remove_tags_from_sent(x) for x in sents_from_html(html_target)]
    source_toks = word_tokenize(' '.join(source_sents))
    target_toks = word_tokenize(' '.join(target_sents))

    source_toks_no_stop = [x for x in source_toks if x.lower() not in stopwords]
    target_toks_no_stop = [x for x in target_toks if x.lower() not in stopwords]
    source_tok_cts = Counter(source_toks_no_stop)
    target_tok_cts = Counter(target_toks_no_stop)

    source_tf_idf = np.zeros([len(idf_df), ])
    source_tf_idf.fill(1e-3)
    target_tf_idf = np.zeros([len(idf_df), ])
    target_tf_idf.fill(1e-3)

    for tok, tf in source_tok_cts.items():
        idf = idf_score[tok]
        idx = tok_to_id.get(tok, 0)
        source_tf_idf[idx] = tf * idf
    for tok, tf in target_tok_cts.items():
        idf = idf_score[tok]
        idx = tok_to_id.get(tok, 0)
        target_tf_idf[idx] = tf * idf

    source_target_similarity = 1 - spatial.distance.cosine(source_tf_idf, target_tf_idf)
    source_vocab = set(source_toks)
    target_vocab = set(target_toks)
    frag_stats = parse_extractive_fragments(source_toks, target_toks)
    stats.update(frag_stats)

    stats['lex_source_target_sim'] = source_target_similarity
    stats['num_notes'] = len(note_types)
    stats['num_note_types'] = len(note_types_cts)

    stats['source_vocab_size'] = len(source_vocab)
    stats['target_vocab_size'] = len(target_vocab)
    stats['vocab_compression'] = len(source_vocab) / max(1, len(target_vocab))

    note_type_dist = [0.0] * len(corpus_note_types)
    for nt_idx, nt in enumerate(corpus_note_types):
        stats[f'{nt}_count'] = note_types_cts[nt]
        note_type_dist[nt_idx] = note_types_cts[nt] / float(len(note_types))

    note_type_entropy = entropy(note_type_dist)
    stats['note_type_entropy'] = note_type_entropy
    stats['source_sents'] = len(source_sents)
    stats['redundancy'] = calculate_redundancy(source_sents)

    stats['source_toks'] = len(source_toks)
    stats['source_sent_len'] = stats['source_toks'] / max(1, stats['source_sents'])
    stats['target_sents'] = len(target_sents)
    stats['target_toks'] = len(target_toks)
    stats['target_sent_len'] = stats['target_toks'] / max(1, stats['target_sents'])
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing Corpus statistics with ACM entities extracted')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('--version', default='original', choices=[
        'original',
        'revised_balanced',
        'revised_max_coverage',
        'revised_extractive',
        'filter_quality',
        'filter_admit',
    ])
    parser.add_argument('--splits', default='validation,train,test', choices=
    ['train', 'validation', 'test', 'train,validation', 'validation,test', 'train,test', 'train,validation,test'])

    args = parser.parse_args()
    data_dir = os.path.join(args.input_dir, args.target)
    corpus_note_types = pd.read_csv(os.path.join(args.input_dir, 'note_types.csv'))['note_type'].dropna().tolist()

    idf_fn = os.path.join(data_dir, 'toks.csv')
    idf_df = pd.read_csv(idf_fn)
    idf_df = idf_df.assign(idx=range(len(idf_df)))
    tok_to_id = dict(zip(idf_df['tok'], idf_df['idx']))
    default_idf = idf_df['idf'].max()
    idf_score = defaultdict(lambda: default_idf, dict(zip(idf_df['tok'], idf_df['idf'])))
    stopwords = STOPWORDS['en']
    stopwords = stopwords.union(set([x for x in punctuation]))

    out_dir = os.path.join(data_dir, 'dataset_stats')
    os.makedirs(out_dir, exist_ok=True)

    ent_dir = os.path.join(data_dir, 'acm_output')
    all_dfs = []
    splits = args.splits.split(',')
    data_suffix = '' if args.version == 'original' else '_' + args.version
    in_fn = os.path.join(args.data_dir, f'summary_dataset{data_suffix}.csv')
    print(f'Loading summary dataset from {in_fn}')
    df = pd.read_csv(in_fn)
    df = add_meta(df, data_dir)
    df = df.sort_values(by='source', key=lambda x: x.str.len())
    for split in splits:
        split_df = df[df['split'] == split]
        out_fn = os.path.join(out_dir, f'{args.version}_{split}_stats.csv')
        records = split_df.to_dict('records')
        if args.cpu_frac == -1:
            out_df = pd.DataFrame(list(tqdm(map(compute_corpus_statistics, records))))
        else:
            out_df = pd.DataFrame(list(p_uimap(compute_corpus_statistics, records)))

        out_df.to_csv(out_fn, index=False)
        all_dfs.append(out_df.assign(split=split))
        cols = [x for x in out_df.columns if x != 'fragments']
        for col in cols:
            valid = out_df[col].dropna()
            try:
                print(f'{col} --> mean={valid.mean()} median={np.median(valid)}')
            except:
                print(f'Cant process {col}')

    all_dfs = pd.concat(all_dfs)
    out_fn = os.path.join(data_dir, 'stats.csv')
    all_dfs.to_csv(out_fn, index=False)
