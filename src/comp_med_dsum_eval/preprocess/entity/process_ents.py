# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from glob import glob
import os
import ujson

import argparse
from gensim.models import KeyedVectors
from nltk import word_tokenize
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm

from comp_med_dsum_eval.eval.rouge import preprocess_sentence, stopwords


def is_match(embed_score, tok_score, code_score, agg_score):
    if embed_score >= 0.75:
        return True
    if code_score is not None and code_score >= 0.4:
        return True
    if agg_score >= 0.4:
        return True
    return False


def compute_overlap_status(start, end, ranges):
    existing_range = (start, end)
    if existing_range in ranges:
        return 'same'
    for range in ranges:
        if start < range[1] and range[0] < end:
            if start >= range[0] and end <= range[1]:
                return 'subset'
            elif start <= range[0] and end >= range[1]:
                return 'superset'
            return 'overlap'
    return 'none'


def filter_ents(ents, link_col='dtype'):
    ent_keep = []
    ranges = []

    icd_ents = list(sorted([x for x in ents if x[link_col] == 'icd'], key=lambda x: (x['BeginOffset'], -x['EndOffset'])))
    rx_ents = list(sorted([x for x in ents if x[link_col] == 'rx'], key=lambda x: (x['BeginOffset'], -x['EndOffset'])))
    ent_ents = list(sorted([x for x in ents if x[link_col] == 'ent'], key=lambda x: (x['BeginOffset'], -x['EndOffset'])))

    sorted_ents = icd_ents + rx_ents + ent_ents
    assert len(sorted_ents) == len(ents)

    for ent in sorted_ents:
        dtype = ent[link_col]
        if dtype == 'icd' and ent['Category'] in {'ANATOMY', 'TIME_EXPRESSION', 'PROTECTED_HEALTH_INFORMATION'}:
            continue
        if dtype == 'ent' and ent['Category'] in {'ANATOMY', 'TIME_EXPRESSION', 'PROTECTED_HEALTH_INFORMATION'}:
            continue
        start, end = ent['BeginOffset'], ent['EndOffset']
        overlap_status = compute_overlap_status(start, end, ranges)
        if overlap_status in {'same', 'superset', 'subset'}:
            continue
        elif overlap_status in {'none', 'overlap'}:
            ent_keep.append(ent)
        else:
            raise Exception(f'Unknown overlap status code = {overlap_status}')
        ranges.append((start, end))

    return list(sorted(ent_keep, key=lambda x: (x['BeginOffset'], -x['EndOffset'])))


def get_icd_codes(ent_obj):
    if ent_obj['dtype'] == 'icd':
        return [(x['Code'], x['Score']) for x in ent_obj['ICD10CMConcepts']]
    else:
        return []


def get_rx_codes(ent_obj):
    if ent_obj['dtype'] == 'rx':
        return [(x['Code'], x['Score']) for x in ent_obj['RxNormConcepts']]
    else:
        return []


def get_icd_code_descriptions(ent_obj):
    if ent_obj['dtype'] == 'icd':
        return [x['Description'] for x in ent_obj['ICD10CMConcepts']]
    else:
        return []


def _compute_code_overlap(a_codes, b_codes):
    a_set = set([x[0] for x in a_codes])
    b_set = set([x[0] for x in b_codes])
    a_denom = sum([x[1] for x in a_codes])
    b_denom = sum([x[1] for x in b_codes])
    a_overlap = sum([x[1] for x in a_codes if x[0] in b_set])
    b_overlap = sum([x[1] for x in b_codes if x[0] in a_set])
    if a_denom + b_denom == 0:
        return 0
    return (a_overlap + b_overlap) / (a_denom + b_denom)


def compute_code_overlap(ent_a, ent_b):
    dtype_a = ent_a['dtype']
    dtype_b = ent_b['dtype']
    rx_ct = sum([1 if dtype == 'rx' else 0 for dtype in [dtype_a, dtype_b]])
    icd_ct = sum([1 if dtype == 'icd' else 0 for dtype in [dtype_a, dtype_b]])

    a_codes, b_codes = [], []
    if icd_ct == 2:
        a_codes, b_codes = get_icd_codes(ent_a), get_icd_codes(ent_b)
    elif rx_ct == 2:
        a_codes, b_codes = get_rx_codes(ent_a), get_rx_codes(ent_b)

    code_overlap = None
    if max(len(a_codes), len(b_codes)) > 0:
        code_overlap = _compute_code_overlap(a_codes, b_codes)
    return code_overlap


def compute_tf_idf_matrix(texts, tf_idf_map, default_idf):
    n = len(texts)
    idf_matrix = np.zeros([n, n])
    toks = list(map(lambda x: [x for x in word_tokenize(x.lower()) if x not in stopwords], texts))
    for i in range(n):
        i_denom = sum([tf_idf_map.get(tok, default_idf) for tok in toks[i]])
        for j in range(i + 1, n):
            overlaps = list(set(toks[i]).intersection(toks[j]))
            overlap_idf = sum([tf_idf_map.get(tok, default_idf) for tok in overlaps])
            j_denom = sum([tf_idf_map.get(tok, default_idf) for tok in toks[j]])
            if i_denom + j_denom == 0:
                idf_matrix[i, j] = idf_matrix[j, i] = 0
            else:
                idf_matrix[i, j] = idf_matrix[j, i] = overlap_idf / (0.5 * i_denom + 0.5 * j_denom)
    return idf_matrix


def compute_lex_sim_matrix(texts, wv):
    n = len(texts)
    embed_matrix = np.zeros([n, n])
    toks = list(map(lambda x: [x for x in word_tokenize(x.lower()) if x not in stopwords], texts))
    for i in range(n):
        for j in range(i + 1, n):
            if min(len(toks[i]), len(toks[j])) == 0:
                embed_matrix[i, j] = embed_matrix[j, i] = 0
            else:
                embed_matrix[i, j] = embed_matrix[j, i] = wv.n_similarity(toks[i], toks[j])
    return embed_matrix


def add_ent_id(ent_obj, source, return_ents=True):
    ent_id_prefix = f'{source}-'
    if ent_obj['dtype'] == 'sec':
        sec_idx = ent_obj['sec_idx']
        ent_id_prefix += f'sec-{sec_idx}-'
    elif ent_obj['dtype'] == 'para':
        para_idx = ent_obj['para_idx']
        ent_id_prefix += f'para-{para_idx}-'
    elif ent_obj['dtype'] == 'sent':
        sent_idx = ent_obj['sent_idx']
        ent_id_prefix += f'sent-{sent_idx}-'
    for i in range(len(ent_obj['ents'])):
        ent_span = str(ent_obj['ents'][i]['BeginOffset']) + '-' + str(ent_obj['ents'][i]['EndOffset'])
        ent_obj['ents'][i]['ent_id'] = ent_id_prefix + ent_span
        ent_obj['ents'][i]['note_idx'] = ent_obj['note_idx']
        ent_obj['ents'][i]['sec_idx'] = ent_obj['sec_idx']
        ent_obj['ents'][i]['sent_idx'] = ent_obj.get('sent_idx', None)
        ent_obj['ents'][i]['text_type'] = ent_obj['dtype']
    if return_ents:
        return ent_obj['ents']
    return ent_obj


def process(fn, wv, tf_idf_map, default_idf, vocab):
    with open(fn, 'r') as fd:
        ents = ujson.load(fd)
    source_ents = ents['source']
    target_ents = ents['target']

    source_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'source')) for ent_obj in source_ents]
    ))

    target_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'target')) for ent_obj in target_ents]
    ))

    source_texts = list(map(lambda x: x['Text'], source_ents_flat))
    target_texts = list(map(lambda x: x['Text'], target_ents_flat))
    source_texts_cleaned = list(map(lambda x: preprocess_sentence(x, vocab_filter=vocab), source_texts))
    target_texts_cleaned = list(map(lambda x: preprocess_sentence(x, vocab_filter=vocab), target_texts))

    source_toks = [x.split(' ') for x in source_texts_cleaned]
    target_toks = [x.split(' ') for x in target_texts_cleaned]

    merge_scores = []
    for target_idx, target_ent_obj in enumerate(target_ents_flat):
        tok_t = [x for x in target_toks[target_idx] if len(x) > 0]
        target_denom = sum([tf_idf_map.get(tok, default_idf) for tok in tok_t])
        for source_idx, source_ent_obj in enumerate(source_ents_flat):
            tok_s = [x for x in source_toks[source_idx] if len(x) > 0]
            source_denom = sum([tf_idf_map.get(tok, default_idf) for tok in tok_s])
            overlaps = list(set(tok_t).intersection(tok_s))
            overlap_idf = sum([tf_idf_map.get(tok, default_idf) for tok in overlaps])
            zero_denom = (target_denom + source_denom) == 0
            tok_overlap = 0 if zero_denom else overlap_idf / (0.5 * source_denom + 0.5 * target_denom)
            lex_sim = 0 if min(len(tok_s), len(tok_t)) == 0 else wv.n_similarity(tok_s, tok_t)

            code_overlap = compute_code_overlap(target_ent_obj, source_ent_obj)

            if code_overlap is None:
                agg_score = (tok_overlap + lex_sim) / 2.0
            else:
                agg_score = (code_overlap + tok_overlap + lex_sim) / 3.0

            should_merge = is_match(lex_sim, tok_overlap, code_overlap, agg_score)
            if should_merge:
                merge_scores.append({
                    'source_ent_id': source_ent_obj['ent_id'],
                    'target_ent_id': target_ent_obj['ent_id'],
                    'source_text': source_ent_obj['Text'],
                    'target_text': target_ent_obj['Text'],
                    'code_overlap': code_overlap,
                    'tok_overlap': tok_overlap,
                    'lex_sim': lex_sim,
                    'agg_score': agg_score,
                    'should_merge': should_merge,
                })
    merge_scores = pd.DataFrame(merge_scores)
    out_fn = fn.replace('json', 'csv')
    merge_scores.to_csv(out_fn, index=False)
    return 1


def safe_mean(arr):
    return np.mean(list(filter(None, arr)))


def get_vocab_info(input_dir):
    kv_fn = os.path.expanduser('/efs/griadams/biovec/BioWordVec_PubMed_MIMICIII_d200.kv')
    bin_fn = os.path.expanduser('/efs/griadams/biovec/BioWordVec_PubMed_MIMICIII_d200.vec.bin')
    if os.path.exists(kv_fn):
        print(f'Loading W2V model from {kv_fn}...')
        wv = KeyedVectors.load(kv_fn, mmap='r')
    else:
        print(f'Loading W2V model from {bin_fn}...')
        wv = KeyedVectors.load_word2vec_format(bin_fn, binary=True)
        wv.fill_norms()
        print(f'Saving vectors in kv format to {kv_fn}')
        wv.save(kv_fn)
    vocab = set(wv.index_to_key)
    print(f'W2V vocabulary of size {len(vocab)}')

    tf_idf = os.path.join(input_dir, 'toks_uncased.csv')
    tf_idf_df = pd.read_csv(tf_idf)
    default_idf = tf_idf_df['idf'].dropna().max()
    tf_idf_map = dict(zip(tf_idf_df['tok'], tf_idf_df['idf']))
    return wv, tf_idf_map, default_idf, vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merging extracted entities between source and target')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)
    wv, tf_idf_map, default_idf, vocab = get_vocab_info(data_dir)

    ent_dir = os.path.join(data_dir, 'acm_output')

    pattern = ent_dir + '/*.json'
    print('Searching for entities...')
    fns = glob(pattern)
    if args.cpu_frac == -1:
        statuses = list(tqdm(map(lambda fn: process(fn, wv, tf_idf_map, default_idf, vocab), fns)))
    else:
        statuses = list(p_uimap(lambda fn: process(fn, wv, tf_idf_map, default_idf, vocab), fns))
    print(f'Pre-computed merge scores for {sum(statuses)} examples')
