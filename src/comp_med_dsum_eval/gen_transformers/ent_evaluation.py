# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ujson

import argparse
import pandas as pd
from p_tqdm import p_uimap
import regex as re
from tqdm import tqdm


from comp_med_dsum_eval.preprocess.entity.process_ents import filter_ents
from comp_med_dsum_eval.perturber.process_ents_for_perturbed import flatten_ents


def annotate(sum_ents, sum_merge_scores):
    if sum_merge_scores is None:
        source_sum_overlap_ent_ids = set()
        target_sum_overlap_ent_ids = set()
    else:
        sum_merge_scores = sum_merge_scores[sum_merge_scores['should_merge']]
        source_sum_scores = sum_merge_scores[sum_merge_scores['relation'] == 'source-sum']
        target_sum_scores = sum_merge_scores[sum_merge_scores['relation'] == 'target-sum']
        source_sum_overlap_ent_ids = set(source_sum_scores['sum_ent_id'].tolist())
        target_sum_overlap_ent_ids = set(target_sum_scores['sum_ent_id'].tolist())

    annotated = ''
    sum_ents_flat = filter_ents(flatten_ents(sum_ents['ents']))
    prev_start = 0
    num_global, num_local = 0, 0
    for ent in sum_ents_flat:
        start, end = ent['BeginOffset'], ent['EndOffset']
        prev_chunk = sum_ents['text'][prev_start:start]
        prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
        annotated += prev_chunk.rstrip()
        category = ent['Category']
        type = ent['Type']
        ent_id = ent['ent_id']
        global_halluc = 0 if ent['ent_id'] in source_sum_overlap_ent_ids else 1
        local_halluc = 0 if ent['ent_id'] in target_sum_overlap_ent_ids else 1
        ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
        annotated += f'<e local={local_halluc} global={global_halluc} cat={category} type={type} id={ent_id}>' \
                     f'{ent_space_pre}' + sum_ents['text'][start:end] + '</e>'
        prev_start = end
        num_global += global_halluc
        num_local += local_halluc
    annotated += sum_ents['text'][prev_start:]
    return {
        'pred_num_ents': len(sum_ents_flat),
        'pred_local_halluc': num_local,
        'pred_global_halluc': num_global,
        'pred_annotated': annotated
    }


def process(record, data_dir, experiment_dir):
    example_id = record['example_id']
    ent_fn = os.path.join(experiment_dir, 'acm_output', f'{example_id}.json')
    merge_fn = os.path.join(experiment_dir, 'ent_merges', f'{example_id}.csv')
    try:
        sum_merges = pd.read_csv(merge_fn)
        sum_merges = sum_merges[sum_merges['should_merge']]
    except:
        sum_merges = None
    with open(ent_fn, 'r') as fd:
        sum_ents = ujson.load(fd)

    try:
        reg_merges = pd.read_csv(os.path.join(data_dir, 'acm_output', f'{example_id}.csv'))
        reg_merges = reg_merges[reg_merges['should_merge']]
        target_ent_ids_non_halluc = reg_merges['target_ent_id'].unique()
    except:
        print('Not picking up merge file.')
        target_ent_ids_non_halluc = []
    annotate_info = annotate(sum_ents, sum_merges)

    num_target_non_halluc = len(target_ent_ids_non_halluc)
    try:
        sum_covered_target_ent_ids = sum_merges['target_ent_id'].unique().tolist()
    except:
        print('No matching summary-reference entities.')
        sum_covered_target_ent_ids = []
    sum_covered_target_ents = set(target_ent_ids_non_halluc).intersection(set(sum_covered_target_ent_ids))
    num_covered_non_halluc = len(sum_covered_target_ents)

    record['num_target_source_covered'] = num_target_non_halluc
    record['num_target_sum_source_covered'] = num_covered_non_halluc
    faith_ent_recall = 1 if num_target_non_halluc == 0 else num_covered_non_halluc / num_target_non_halluc
    record['faithful_ent_recall'] = faith_ent_recall
    assert faith_ent_recall >= 0 and faith_ent_recall <= 1

    record.update(annotate_info)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate revised outputs for entity overlap with original reference and summary')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default=None, required=True)
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''
    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)
    experiment_dir = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment)
    gen_fn = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment, 'outputs.csv')
    gen_df = pd.read_csv(gen_fn)
    records = gen_df.to_dict('records')

    if args.cpu_frac == -1:
        annotated_df = pd.DataFrame(list(tqdm(map(
            lambda record: process(record, data_dir, experiment_dir), records), total=len(records))))
    else:
        annotated_df = pd.DataFrame(list(p_uimap(
            lambda record: process(record, data_dir, experiment_dir), records)))
    print(f'Successfully processed {len(annotated_df)} examples')

    out_fn = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment, 'outputs_annotated.csv')
    print(f'Saving examples to {out_fn}')
    annotated_df.to_csv(out_fn, index=False)

    halluc_cols = ['pred_num_ents', 'pred_local_halluc', 'pred_global_halluc', 'faithful_ent_recall']
    for col in halluc_cols:
        mean = annotated_df[col].dropna().mean()
        print(f'{col},{mean}')
