# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from glob import glob
import itertools
import json
import os
import spacy
import pickle
import ujson

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from p_tqdm import p_uimap
import regex as re
from tqdm import tqdm

from comp_med_dsum_eval.preprocess.sec_tag.section_utils import get_sent_idxs, remove_tags_from_sent, sents_from_html
from comp_med_dsum_eval.preprocess.entity.entity_utils import extract_ents, extract_ent_ids
from comp_med_dsum_eval.preprocess.entity.entity_utils import annotate_sent_w_ents, annotate_sent_w_ents_add_halluc
from comp_med_dsum_eval.preprocess.entity.process_ents import add_ent_id
from comp_med_dsum_eval.ref_reviser.build_train_dataset_from_perturbed import (
    embeds_by_sent_idx, add_to_embeds, get_idxs_from_str, nums_from_str, numpy_to_json_dict
)


def retrieve_context(source_sent_map, source_embeds, target_embed_obj, max_retrievals=5):
    added_sents = []
    added_sent_idxs = []
    added_priors = []
    added_scores = []
    target_n = min(len(source_embeds), max_retrievals)
    used_ids = []
    improvements = []
    max_coverages = np.zeros(shape=(len(target_embed_obj['h_no_stop']), ))
    sims = [
        np.clip(cosine_similarity(
            target_embed_obj['h_no_stop'], x['bert_h']).max(axis=1), 0, 1) for x in source_embeds
    ]
    scores_prior = [sim.mean() for sim in sims]
    while len(added_sents) < target_n:
        weights = np.clip(1 - max_coverages, 1e-4, 1)
        scores = [(sim * weights).sum() / weights.sum() for sim in sims]
        for id in used_ids:  # Don't reuse the same sentence
            scores[id] = float('-inf')
        max_score = np.max(scores)
        best_id = int(np.argmax(scores))
        used_ids.append(best_id)
        best_sent_idx = source_embeds[best_id]['sent_idx']
        added_sent_idxs.append(best_sent_idx)
        added_sents.append(source_sent_map[best_sent_idx])
        new_max_coverages = np.maximum(max_coverages, sims[best_id])
        max_improvement = (new_max_coverages - max_coverages).max()
        mean_improvement = (new_max_coverages - max_coverages).mean()
        improvements.append((mean_improvement, max_improvement))
        added_priors.append(scores_prior[best_id])
        added_scores.append(max_score)
        max_coverages = new_max_coverages
    return added_sents, added_sent_idxs, max_coverages, added_scores, added_priors, improvements


def process(data_dir, out_dir, example, nlp, max_context_sents=10):
    dataset_outputs = []

    source = example['source']
    source_sents = sents_from_html(source)
    source_idxs = get_sent_idxs(example['source'])
    assert len(source_idxs) == len(source_sents)

    used_sents = set()
    used_sent_map = {}
    for sent_idx, source_sent in zip(source_idxs, source_sents):
        if source_sent.lower() in used_sents:
            continue
        used_sents.add(source_sent.lower())
        used_sent_map[sent_idx] = source_sent
    used_sent_idxs = set(used_sent_map.keys())
    target = example['target']
    target_sents = sents_from_html(target)
    target_sent_idxs = get_sent_idxs(example['target'])
    example_id = example['example_id']
    regular_ent_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.json')
    regular_ent_merge_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.csv')

    with open(regular_ent_fn) as fd:
        reg_ents = ujson.load(fd)

    source_ents = reg_ents['source']
    target_ents = reg_ents['target']

    try:
        reg_ent_merges = pd.read_csv(regular_ent_merge_fn)
        reg_ent_merges = reg_ent_merges[reg_ent_merges['should_merge']]
        reg_ent_merges = reg_ent_merges[reg_ent_merges['source_ent_id'].apply(lambda x: 'sent' in x)]
        if len(reg_ent_merges) == 0:
            reg_ent_merges = None
    except:
        reg_ent_merges = None

    with open(os.path.join(data_dir, 'embed_cache', f'{example_id}.pk'), 'rb') as fd:
        embeds = pickle.load(fd)

    source_embeds = [embed for embed in embeds['source'] if embed['sent_idx'] in used_sent_idxs]

    num_outputs = []
    for target_sent_idx in target_sent_idxs:
        target_embed_obj = [x for x in embeds['target'] if x['sent_idx'] == target_sent_idx]
        assert len(target_embed_obj) == 1
        target_embed_obj = target_embed_obj[0]
        target_ent_obj = [x for x in target_ents if x['dtype'] == 'sent' and x['sent_idx'] == target_sent_idx]
        assert len(target_ent_obj) == 1
        target_ent_obj = target_ent_obj[0]
        add_to_embeds(target_embed_obj, tok_col='bert_token', h_col='bert_h')
        target_ent_obj = add_ent_id(target_ent_obj, 'target', return_ents=False)
        if len(target_embed_obj['h_no_special']) < len(target_embed_obj['bert_token']):
            continue

        if reg_ent_merges is None:
            merges_by_ent = {}
        else:
            reg_sent_merges = reg_ent_merges[reg_ent_merges['target_ent_id'].apply(
                lambda x: f'sent-{target_sent_idx}-' in x)]
            merges_by_ent = dict(tuple(reg_sent_merges.groupby('target_ent_id')))
        target_sent_annotated = annotate_sent_w_ents_add_halluc(target_ent_obj, set(merges_by_ent.keys()))
        target_sent_clean = remove_tags_from_sent(target_sent_annotated)
        target_nums = nums_from_str(target_sent_clean, nlp)

        _, _, _, _, target_ent_strs = extract_ents(target_sent_annotated, {})
        target_ent_ids = extract_ent_ids(target_sent_annotated)
        target_ent_strs_lower = [x.lower() for x in target_ent_strs]
        ent_bert_idxs = [get_idxs_from_str(str, target_embed_obj['bert_token']) for str in target_ent_strs_lower]
        ent_h = []
        for bert_idx_set in ent_bert_idxs:
            if len(bert_idx_set) == 0:
                raise Exception(f'Cannot locate an entity in example={example_id}, sent={target_sent_idx}')
            ent_h.append(target_embed_obj['h_no_special'][bert_idx_set, :].mean(axis=0))
        target_ent_source_sent_idxs = []
        target_ent_id2ent_idx = {}
        for k, v in merges_by_ent.items():
            source_sent_idxs = [int(re.search(r'source-sent-(\d+)-', x).group(1))
                                for x in v['source_ent_id'].dropna().tolist()]
            avail_source_sent_idxs = [x for x in source_sent_idxs if x in used_sent_idxs]
            target_ent_source_sent_idxs.append((k, avail_source_sent_idxs))
            target_ent_id2ent_idx[k] = target_ent_ids.index(k)
        mergeable_target_ents = len(target_ent_id2ent_idx)
        num_target_ents = len(re.findall('<e', target_sent_annotated))
        num_hallucinations = num_target_ents - mergeable_target_ents
        added_sents, added_sent_idxs, max_coverages, added_scores, added_priors, improvements = retrieve_context(
            used_sent_map, source_embeds, target_embed_obj)
        covered_ents = 0
        missing_target_ent_ids = []
        sent_idx2target_ent_id = defaultdict(list)
        for target_ent_id, source_sent_ids in target_ent_source_sent_idxs:
            overlapping_sent_idxs = set(source_sent_ids).intersection(set(added_sent_idxs))
            if len(overlapping_sent_idxs) == 0:
                missing_sent_ids = list(set(source_sent_ids) - set(added_sent_idxs))
                for sent_id in missing_sent_ids:
                    sent_idx2target_ent_id[sent_id].append(target_ent_id)
                missing_target_ent_ids.append(target_ent_id)
            else:
                covered_ents += 1

        recovered_target_ent_ids = set()
        if len(sent_idx2target_ent_id) > 0:
            num_extractions = 0
            max_num_extraction = max(1, max_context_sents // 2)
            while len(recovered_target_ent_ids) < len(missing_target_ent_ids) and num_extractions < max_num_extraction:
                remaining_target_ent_ids = set(missing_target_ent_ids) - recovered_target_ent_ids
                sent_set = set()
                for target_ent_id, sent_list in target_ent_source_sent_idxs:
                    if target_ent_id in remaining_target_ent_ids:
                        for sent_idx in sent_list:
                            sent_set.add(sent_idx)
                if len(sent_set) == 0:
                    break
                missing_ent_idxs = [target_ent_id2ent_idx[x] for x in list(remaining_target_ent_ids)]
                missing_h = np.stack([ent_h[x] for x in missing_ent_idxs])
                candidate_source_embeds = [embed for embed in embeds['source'] if embed['sent_idx'] in sent_set]
                if len(candidate_source_embeds) == 0:
                    # TODO figure out why we can't find the >0 sent_set sentences in embeds[source]
                    print(f'Cannot find sentences {sent_set} in embeds[source] for example={example_id}')
                    break
                sims = [
                    np.clip(cosine_similarity(missing_h, x['bert_h']).max(axis=1), 0, 1) for x in candidate_source_embeds
                ]
                scores = [sim.max() for sim in sims]
                best_id = int(np.argmax(scores))
                best_sent_idx = candidate_source_embeds[best_id]['sent_idx']
                added_sent_idxs.append(best_sent_idx)
                improvements.append(-1)
                added_sents.append(used_sent_map[best_sent_idx])
                for target_ent_id in sent_idx2target_ent_id[best_sent_idx]:
                    recovered_target_ent_ids.add(target_ent_id)
                num_extractions += 1

        added_embeds = []
        for sent_idx in added_sent_idxs:
            source_embed_obj = embeds_by_sent_idx(embeds['source'], sent_idx)
            add_to_embeds(source_embed_obj, tok_col='bert_token', h_col='bert_h')
            added_embeds.append(source_embed_obj)
            source_rel_to_target = [
                float(cosine_similarity(np.expand_dims(x, axis=0), target_embed_obj['h_no_special'])[0].max())
                for x in source_embed_obj['word_states']]
            source_embed_obj['source_rel_to_target'] = source_rel_to_target

        total_covered_ents = len(recovered_target_ent_ids) + covered_ents

        added_h_no_stop_flat = np.concatenate([x['h_no_stop'] for x in added_embeds], axis=0)
        target_source_sims_no_stop = cosine_similarity(target_embed_obj['h_no_stop'], added_h_no_stop_flat)
        source_to_target_coverage = float(target_source_sims_no_stop.max(axis=1).mean())
        target_to_source_coverage = float(target_source_sims_no_stop.max(axis=0).mean())
        sent_level_sims = [cosine_similarity(target_embed_obj['h_no_stop'], h['h_no_stop']) for h in added_embeds]
        source_sents_to_target_coverage = [float(x.max(axis=1).mean()) for x in sent_level_sims]
        target_to_source_sents_coverage = [float(x.max(axis=0).mean()) for x in sent_level_sims]

        overlaps = {
            'source_to_target_coverage': source_to_target_coverage,
            'target_to_source_coverage': target_to_source_coverage,
        }
        target_embed_obj.update(overlaps)

        for added_idx in range(len(added_embeds)):
            added_embeds[added_idx]['source_to_target_coverage'] = source_sents_to_target_coverage[added_idx]
            added_embeds[added_idx]['target_to_source_coverage'] = target_to_source_sents_coverage[added_idx]

        added_nums = nums_from_str(' '.join(added_sents), nlp)
        missing_num_set = set(target_nums) - set(added_nums)
        missing_nums = list(missing_num_set)
        covered_nums = len(target_nums) - len(missing_nums)

        assert len(added_sent_idxs) == len(set(added_sent_idxs))
        num_source_toks = len(' '.join(added_sents).split(' '))
        assert total_covered_ents <= mergeable_target_ents <= num_target_ents

        annotated_source = []
        for sent_idx in added_sent_idxs:
            sent_ents = [
                x for x in source_ents if 'sent_idx' in x and x['sent_idx'] == sent_idx and x['dtype'] == 'sent']
            assert len(sent_ents) == 1
            sent_ent_obj = add_ent_id(sent_ents[0], 'source', return_ents=False)
            annotated_source.append(annotate_sent_w_ents(sent_ent_obj))

        # rel_merges = []
        # sent_prefixes = [f'source-sent-{sent_idx}-' for sent_idx in added_sent_idxs]
        # if reg_sent_merges is not None:
        #     for record in reg_sent_merges.to_dict('records'):
        #         if any([record['source_ent_id'].startswith(p) for p in sent_prefixes]):
        #             record['relation'] = 'source-target'
        #             rel_merges.append(record)

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

        source_alignments = [numpy_to_json_dict(e, k_delete=remove_keys) for e in added_embeds]
        target_alignments = numpy_to_json_dict(target_embed_obj, k_delete=remove_keys)

        stats = {
            'target_ent_num': num_target_ents,
            'source_linked_target_ent_num': mergeable_target_ents,
            'num_hallucinations': num_hallucinations,
            'covered_ent_num': total_covered_ents,
            'source_sents': len(added_sent_idxs),
            'source_to_target_coverage': target_embed_obj['source_to_target_coverage'],
            'target_to_source_coverage': target_embed_obj['target_to_source_coverage'],
            'source_toks': num_source_toks,
            'target_nums': len(target_nums),
            'covered_nums': covered_nums,
        }

        dataset_output = {
            'example_id': example_id,
            'source': {
                'sent_idxs': added_sent_idxs,
                'num_tokens': num_source_toks,
                'alignment': source_alignments,
                'sents': annotated_source,
                'improvements': improvements,
            },
            'target': {
                'alignment': target_alignments,
                'sent_idx': target_sent_idx,
                'previous_sent': '' if target_sent_idx == 0 else target_sents[target_sent_idx - 1],
                'sent': target_sent_annotated,
            },
            'stats': stats
        }

        dataset_outputs.append(dataset_output)
        num_outputs.append(stats)
    out_fn = os.path.join(out_dir, f'{example_id}.json')
    with open(out_fn, 'w') as fd:
        json.dump(dataset_outputs, fd)
    return num_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Retrieving relevant sentences from source to create improved dataset.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('--max_n', default=None, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)
    mini_str = '_mini' if args.debug else ''

    out_dir = os.path.join(data_dir, 'revise', 'contexts')
    os.makedirs(out_dir, exist_ok=True)

    print('Loading SciSpacy')
    nlp = spacy.load('en_core_sci_sm')

    fn = os.path.join(data_dir, f'summary_dataset{mini_str}.csv')
    print(f'Reading dataset from {fn}')
    examples = pd.read_csv(fn)

    print(f'Loaded {len(examples)} examples')
    print(f'{len(examples)} ready to be processed.')
    if args.only_new:
        done_example_ids = set([x.split('/')[-1].replace('.json', '') for x in glob(out_dir + '/*.json')])
        examples = examples[~examples['example_id'].isin(done_example_ids)]
        print(f'Filtered out {len(done_example_ids)} already done examples.  Left with {len(examples)}')

    if args.max_n is not None and args.max_n < len(examples):
        examples = examples[:args.max_n]
        print(f'Taking first {args.max_n} examples')
    examples = examples.sort_values(by='source', key=lambda x: x.str.len())
    examples = examples.to_dict('records')

    if args.cpu_frac == -1:
        num_outputs = list(itertools.chain(*list(tqdm(map(
            lambda ex: process(data_dir, out_dir, ex, nlp), examples), total=len(examples)))))
    else:
        num_outputs = list(itertools.chain(*list(p_uimap(
            lambda ex: process(data_dir, out_dir, ex, nlp), examples, num_cpus=args.cpu_frac))))

    num_outputs = pd.DataFrame(num_outputs)
    for col in num_outputs.columns:
        print(col, num_outputs[col].dropna().mean())
