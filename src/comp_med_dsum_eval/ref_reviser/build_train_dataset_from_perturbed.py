# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import itertools
import os
import pickle
from string import punctuation
import ujson

import argparse
import pandas as pd
import numpy as np
np.random.seed(1992)
from sklearn.metrics.pairwise import cosine_similarity
from p_tqdm import p_uimap
import regex as re
from tqdm import tqdm
from lexrank import STOPWORDS
stopwords = STOPWORDS['en']
stopwords = stopwords.union(set([x for x in punctuation]))
from comp_med_dsum_eval.ref_reviser.alignments.utils import keep_sent


def numpy_to_json_dict(obj, k_delete=None):
    new_obj = {}
    for k, v in obj.items():
        if k_delete is not None and k in k_delete:
            continue
        if type(v) == np.ndarray:
            v = v.tolist()
        elif type(v) == list and len(v) > 0 and type(v[0]) == np.ndarray:
            v = [x.tolist() for x in v]
        new_obj[k] = v
    return new_obj


def embeds_by_sent_idx(embeds, sent_idx):
    return [x for x in embeds if x['sent_idx'] == sent_idx][0]


def is_stopword(toks, i):
    n = len(toks)
    try:
        if i < n - 1 and toks[i + 1].startswith('#'):
            return False
    except IndexError:
        print(toks, len(toks), i, n - 1)
        raise
        return False
    return toks[i].lower() in stopwords


def stopword_idxs(toks):
    return [i for i in range(len(toks)) if is_stopword(toks, i)]


def filter_keep_idxs(toks, h):
    assert len(h) <= len(toks)
    idxs = [i for i in range(min(len(toks), len(h))) if toks[i].lower() not in stopwords
            or (i < len(toks) - 1 and toks[i + 1].startswith('#'))]
    hidden_states = h[idxs] if len(idxs) > 0 else h
    return hidden_states, idxs


def nums_from_str(sent_str, nlp):
    return list(set(list(map(str, list(filter(lambda x: x.like_num, nlp(sent_str)))))))


def get_word_idxs(toks, max_idx):
    words = []
    word_idx = []
    curr_word_idx = []
    curr_word = ''
    for i, tok in enumerate(toks):
        if i > max_idx:
            break
        if tok.startswith('#'):
            curr_word_idx.append(i)
            curr_word += tok.lstrip('#')
        else:
            if len(curr_word_idx) > 0:
                word_idx.append(curr_word_idx)
                words.append(curr_word)
            curr_word = tok
            curr_word_idx = [i]
    if len(curr_word_idx) > 0:
        word_idx.append(curr_word_idx)
        words.append(curr_word)
    return words, word_idx


def get_idxs_from_str(str, toks):
    outputs = []
    str = str.replace(' ', '')
    matching_idxs = []
    curr_start = 0
    for tok_idx, tok in enumerate(toks):
        tok_clean = re.escape(tok.lstrip('#'))
        match = re.match(tok_clean, str[curr_start:])
        if match is None:
            if len(matching_idxs) > 0:
                outputs.append(matching_idxs)
                matching_idxs = []
                curr_start = 0
        else:
            matching_idxs.append(tok_idx)
            curr_start += len(tok_clean)
    if len(matching_idxs) > 0:
        outputs.append(matching_idxs)
    if len(outputs) == 0:
        return []
    lens = [len(x) for x in outputs]
    return outputs[np.argmax(lens)]


def word_states(bert_h, bert_word_idxs):
    return [bert_h[i, :].mean(axis=0) for i in bert_word_idxs]


def delete(arr, idxs_to_delete, axis=0):
    valid_idxs_to_delete = [i for i in idxs_to_delete if i < len(arr)]
    return np.delete(arr, valid_idxs_to_delete, axis=axis)


def add_to_embeds(embed_obj, tok_col='bert_tokens', h_col='bert_hidden_states'):
    embed_obj['stopword_idxs'] = stopword_idxs(embed_obj[tok_col])
    embed_obj['h_no_special'] = embed_obj[h_col][1:-1]
    words, word_idxs = get_word_idxs(embed_obj[tok_col], max_idx=len(embed_obj['h_no_special']) - 1)
    embed_obj['words'] = words
    embed_obj['bert_idxs'] = word_idxs
    embed_obj['word_states'] = word_states(embed_obj['h_no_special'], word_idxs)
    h_no_stop = embed_obj['h_no_special']
    if len(embed_obj['stopword_idxs']) < len(embed_obj['h_no_special']):
        h_no_stop = delete(embed_obj['h_no_special'], embed_obj['stopword_idxs'])
    embed_obj['h_no_stop'] = h_no_stop
    return h_no_stop


def process(data_dir, noise_experiment, examples, example_id, out_dir):
    outputs = []
    rel_examples = [ex for ex in examples if ex['example_id'] == example_id]
    perturb_annotated_fn = os.path.join(data_dir, 'perturb', noise_experiment, 'annotated', f'{example_id}.csv')
    annotated_sents = pd.read_csv(perturb_annotated_fn)
    annotated_sents.sort_values(by=['sent_idx', 'perturb_idx'], inplace=True)

    perturb_embed_fn = os.path.join(data_dir, 'perturb', noise_experiment, 'eval', f'{example_id}.pk')
    with open(perturb_embed_fn, 'rb') as fd:
        all_perturb_embeds = pickle.load(fd)

    annotated_sents = annotated_sents.assign(
        span_remove=annotated_sents['masked_str'].apply(lambda x: int(re.search(r'<span-remove-(\d+)>', x).group(1))),
        ent_add=annotated_sents['masked_str'].apply(lambda x: int(re.search(r'<ent-add-(\d+)>', x).group(1))),
        ent_remove=annotated_sents['masked_str'].apply(lambda x: int(re.search(r'<ent-remove-(\d+)>', x).group(1))),
        shuffle=annotated_sents['masked_str'].apply(lambda x: int(re.search(r'<shuffle-(\d+)>', x).group(1))),
    )

    target_sent_idxs = annotated_sents['sent_idx'].unique().tolist()
    for target_sent_idx in target_sent_idxs:
        rel_annotated = annotated_sents[annotated_sents['sent_idx'] == target_sent_idx]
        rel_example = [ex for ex in rel_examples if ex['target']['sent_idx'] == target_sent_idx][0]
        perturb_embeds = [x for x in all_perturb_embeds if x['sent_idx'] == target_sent_idx and '_' in x['key']]
        perturb_embeds = list(sorted(perturb_embeds, key=lambda x: x['perturb_idx']))
        target_embed_obj_plus = [x for x in all_perturb_embeds if x['sent_idx'] == target_sent_idx and '_' not in x['key']]
        target_embed_obj = target_embed_obj_plus[0]
        output = _process(data_dir, rel_example, rel_annotated, perturb_embeds, target_embed_obj)
        if output is not None:
            outputs.append(output)

    out_fn = os.path.join(out_dir, f'{example_id}.json')
    with open(out_fn, 'w') as fd:
        ujson.dump(outputs, fd)
    return outputs


def _process(data_dir, example, annotated_sents, perturb_embeds, target_embed_obj):
    source = example['source']
    example_id = example['example_id']

    with open(os.path.join(data_dir, 'embed_cache', f'{example_id}.pk'), 'rb') as fd:
        embeds = pickle.load(fd)

    source_embeds = embeds['source']
    add_to_embeds(target_embed_obj, tok_col='bert_tokens', h_col='bert_hidden_states')

    if len(target_embed_obj['h_no_special']) < len(target_embed_obj['bert_tokens']):
        uid = example_id + '.' + example['target']['sent_idx']
        print(f'Sentence={uid} is too long.')
        return None

    added_embeds = []
    for sent_idx in source['sent_idxs']:
        source_embed_obj = embeds_by_sent_idx(source_embeds, sent_idx)
        add_to_embeds(source_embed_obj, tok_col='bert_token', h_col='bert_h')
        added_embeds.append(source_embed_obj)

    added_h_no_stop_flat = np.concatenate([x['h_no_stop'] for x in added_embeds], axis=0)

    for perturb_embed in perturb_embeds:
        add_to_embeds(perturb_embed)
        perturb_source_sims_no_stop = cosine_similarity(perturb_embed['h_no_stop'], added_h_no_stop_flat)
        source_to_perturb_coverage = float(perturb_source_sims_no_stop.max(axis=1).mean())
        perturb_to_source_coverage = float(perturb_source_sims_no_stop.max(axis=0).mean())
        sent_level_sims = [cosine_similarity(perturb_embed['h_no_stop'], h['h_no_stop']) for h in added_embeds]
        source_sents_to_perturb_coverage = [float(x.max(axis=1).mean()) for x in sent_level_sims]
        perturb_to_source_sents_coverage = [float(x.max(axis=0).mean()) for x in sent_level_sims]

        added_h_no_special = np.concatenate([x['h_no_special'] for x in added_embeds], axis=0)
        perturb_rel_to_source = [float(
            cosine_similarity(np.expand_dims(x, axis=0), added_h_no_special)[0].max())
            for x in perturb_embed['word_states']]
        perturb_rel_to_target = [
            float(cosine_similarity(np.expand_dims(x, axis=0), target_embed_obj['h_no_special'])[0].max())
            for x in perturb_embed['word_states']
        ]

        perturb_target_sims_no_stop = cosine_similarity(perturb_embed['h_no_stop'], target_embed_obj['h_no_stop'])
        target_to_perturb_coverage = float(perturb_target_sims_no_stop.max(axis=1).mean())
        perturb_to_target_coverage = float(perturb_target_sims_no_stop.max(axis=0).mean())

        for align_idx in range(len(added_embeds)):
            source_rel_to_perturbed = [float(
                cosine_similarity(np.expand_dims(x, axis=0), perturb_embed['h_no_special'])[0].max())
                for x in added_embeds[align_idx]['word_states']]
            overlap = {
                'source_to_perturb_coverage': source_sents_to_perturb_coverage[align_idx],
                'perturb_to_source_coverage': perturb_to_source_sents_coverage[align_idx],
                'source_rel_to_perturbed': source_rel_to_perturbed,
            }
            example['source']['alignment'][align_idx].update(overlap)

        overlaps = {
            'source_to_perturb_coverage': source_to_perturb_coverage,
            'perturb_to_source_coverage': perturb_to_source_coverage,
            'perturb_to_target_coverage': perturb_to_target_coverage,
            'target_to_perturb_coverage': target_to_perturb_coverage,
            'perturb_rel_to_source': perturb_rel_to_source,
            'perturb_rel_to_target': perturb_rel_to_target,
        }

        perturb_embed.update(overlaps)

    remove_keys = {
        'h_no_special',
        'h_no_stop',
        'bert_h',
        'bert_hidden_states',
        'electra_hidden_states',
        'electra_h',
        'electra_logits',
        'word_states'
    }

    perturb_alignments = [numpy_to_json_dict(e, k_delete=remove_keys) for e in perturb_embeds]

    avg_source_to_perturb_coverage = np.mean([x['source_to_perturb_coverage'] for x in perturb_embeds])
    avg_perturb_to_source_coverage = np.mean([x['perturb_to_source_coverage'] for x in perturb_embeds])
    avg_perturb_to_target_coverage = np.mean([x['perturb_to_target_coverage'] for x in perturb_embeds])
    avg_target_to_perturb_coverage = np.mean([x['target_to_perturb_coverage'] for x in perturb_embeds])
    min_perturb_to_target_coverage = min([x['perturb_to_target_coverage'] for x in perturb_embeds])
    min_target_to_perturb_coverage = min([x['target_to_perturb_coverage'] for x in perturb_embeds])

    perturb_stats = {
        'avg_local_halluc': float(annotated_sents['local_halluc'].mean()),
        'max_local_halluc': float(annotated_sents['local_halluc'].max()),
        'avg_source_to_perturb_coverage': float(avg_source_to_perturb_coverage),
        'avg_perturb_to_source_coverage': float(avg_perturb_to_source_coverage),
        'avg_perturb_to_target_coverage': float(avg_perturb_to_target_coverage),
        'avg_target_to_perturb_coverage': float(avg_target_to_perturb_coverage),
        'min_perturb_to_target_coverage': min_perturb_to_target_coverage,
        'min_target_to_perturb_coverage': min_target_to_perturb_coverage,
    }

    example['stats'].update(perturb_stats)

    example['noisy_target'] = {
        'sent': annotated_sents['text_annotated'].tolist(),
        'perturb_idx': annotated_sents['perturb_idx'].tolist(),
        'alignment': perturb_alignments,
        'noise_parameters': annotated_sents[[
            'num_ents_original', 'local_halluc', 'global_halluc', 'span_remove', 'ent_add', 'ent_remove', 'shuffle'
        ]].to_dict('records'),
    }

    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Adding hallucinations from perturber to high quality reference revisions'
                                     'and calculating BERT-level & entity overlaps.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--noise_experiment', default='ent_sample')  # CHANGE this if you change perturber model
    parser.add_argument('--cpu_frac', default=0.8, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('--max_n', default=None, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)
    out_dir = os.path.join(data_dir, 'revise', 'examples')
    os.makedirs(out_dir, exist_ok=True)

    fn = os.path.join(data_dir, 'revise', f'high_quality_w_context{mini_str}.json')
    print(f'Loading high quality reference sentence-source context pairs from {fn}')
    with open(fn, 'r') as fd:
        examples = ujson.load(fd)
    print('Removing all examples from the summarization test set.')
    test_example_ids = set(pd.read_csv(os.path.join(data_dir, 'test_example_ids.csv'))['example_id'])
    examples = list(filter(lambda record: record['example_id'] not in test_example_ids, examples))
    context_example_ids = set([x['example_id'] for x in examples])

    perturb_dir = os.path.join(data_dir, 'perturb', args.noise_experiment, 'annotated')
    perturb_fns = glob(perturb_dir + '/*.csv')
    example_ids = [x.split('/')[-1].replace('.csv', '') for x in perturb_fns]
    prev_n = len(example_ids)

    example_ids = [x for x in context_example_ids if x in context_example_ids]
    new_n = len(example_ids)
    if new_n < prev_n:
        print(f'Could not find source contexts for {prev_n - new_n} so we are filtering them out. '
              f'Could just be debug mode.')

    if args.max_n is not None and args.max_n < len(example_ids):
        example_ids = example_ids[:args.max_n]
        print(f'Taking first {args.max_n} examples')

    if args.only_new:
        done_fns = glob(out_dir + '/*.json')
        done_example_ids = set([x.split('/')[-1].replace('.json', '') for x in done_fns])
        print(f'Choosing not to cache merge entities for the {len(done_example_ids)} already completed examples')
        example_ids = [example_id for example_id in example_ids if example_id not in done_example_ids]

    if args.cpu_frac == -1:
        flat_outputs = list(itertools.chain(*list(tqdm(map(
            lambda ex_id: process(data_dir, args.noise_experiment, examples, ex_id, out_dir), example_ids),
            total=len(example_ids)))))
    else:
        flat_outputs = list(itertools.chain(*list(p_uimap(
            lambda ex_id: process(data_dir, args.noise_experiment, examples, ex_id, out_dir), example_ids,
            num_cpus=args.cpu_frac))))

    outputs = []
    print('Collecting into single file...')
    for example in tqdm(flat_outputs, total=len(flat_outputs)):
        example_id = example['example_id']
        src = example['source']
        perturb = example['noisy_target']
        num_source = len(src['sent_idxs'])
        keep_source_idxs = [i for i in range(num_source) if keep_sent(src['improvements'][i])]
        if len(keep_source_idxs) == 0:
            keep_source_idxs = [0]
        source_sent_idxs = [src['sent_idxs'][i] for i in keep_source_idxs]
        source_sents = [src['sents'][i] for i in keep_source_idxs]
        source_improvements = [src['improvements'][i] for i in keep_source_idxs]
        row = {
            'example_id': example_id,
            'target_sent_idx': example['target']['sent_idx'],
            'target_sent': example['target']['sent'],
            'perturb': {
                'sents': perturb['sent'],
                'source_to_perturb_coverage': [x['source_to_perturb_coverage'] for x in perturb['alignment']],
                'perturb_to_source_coverage': [x['perturb_to_source_coverage'] for x in perturb['alignment']],
            },
            'source': {
                'sent_idxs': source_sent_idxs,
                'sents': source_sents,
                'improvements': source_improvements
            }
        }
        row.update(example['stats'])
        outputs.append(row)

    out_fn = os.path.join(data_dir, 'revise', 'dataset.json')
    print(f'Saving {len(outputs)} examples to {out_fn}')
    with open(out_fn, 'w') as fd:
        ujson.dump(outputs, fd)

    output_mini = outputs[:128]
    out_mini_fn = os.path.join(data_dir, 'revise', f'dataset_mini.json')
    print(f'Saving {len(output_mini)} examples to {out_mini_fn}')
    with open(out_mini_fn, 'w') as fd:
        ujson.dump(output_mini, fd)
