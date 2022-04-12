# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os
import itertools
import ujson

import argparse
import pandas as pd
from p_tqdm import p_uimap
import regex as re
from tqdm import tqdm


from comp_med_dsum_eval.preprocess.entity.process_ents import add_ent_id, filter_ents
from comp_med_dsum_eval.perturber.process_ents_for_perturbed import flatten_ents


def annotate(reg_ents, reg_merge_scores, revise_ents, revise_merge_scores, revise_sents):
    target_ents = reg_ents['target']

    target_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'target')) for ent_obj in target_ents]
    ))

    if reg_merge_scores is None:
        reg_overlap_ent_ids = set()
    else:
        reg_merge_scores = reg_merge_scores[reg_merge_scores['should_merge']].dropna(subset=['source_ent_id', 'target_ent_id'])
        reg_overlap_ent_ids = set(reg_merge_scores['target_ent_id'].tolist())

    if revise_merge_scores is None:
        source_revised_overlap_ent_ids = set()
        target_revised_overlap_ent_ids = set()
        target_revised_scores = []
    else:
        revise_merge_scores = revise_merge_scores[revise_merge_scores['should_merge']]
        source_revised_scores = revise_merge_scores[revise_merge_scores['relation'] == 'source-revised']
        target_revised_scores = revise_merge_scores[revise_merge_scores['relation'] == 'target-revised']
        source_revised_overlap_ent_ids = set(source_revised_scores['revised_ent_id'].tolist())
        target_revised_overlap_ent_ids = set(target_revised_scores['revised_ent_id'].tolist())
        target_revised_scores = target_revised_scores.to_dict('records')

    annotations = {}
    ent_info = {}
    for sent_idx in revise_ents:
        original_target_sent_ents = [
            x for x in target_ents_flat if x['sent_idx'] == int(sent_idx) and x['text_type'] == 'sent']
        if reg_merge_scores is None or len(reg_merge_scores) == 0:
            faithful_orig_target_ids = []
        else:
            reg_sent_scores = reg_merge_scores[reg_merge_scores['target_ent_id'].apply(lambda x: f'sent-{sent_idx}-' in x)]
            faithful_orig_target_ids = [] if len(reg_sent_scores) == 0 else reg_sent_scores['target_ent_id'].unique().tolist()
        original_num_faithful_ents = len(faithful_orig_target_ids)
        original_text = [x['text'] for x in target_ents if x['dtype'] == 'sent' and x['sent_idx'] == int(sent_idx)]
        assert len(original_text) == 1
        original_text = original_text[0]
        revised_objs = revise_ents[sent_idx]['revised']
        for revise_idx, revised_ents in revised_objs.items():
            annotated = ''
            noise_ents_sent = filter_ents(flatten_ents(revised_ents['ents']))
            prev_start = 0
            num_global, num_local = 0, 0
            covered_target_ent_ids = set()
            match_revised_ent_id = f'sent-{sent_idx}-revise-{revise_idx}-'
            for record in target_revised_scores:
                if match_revised_ent_id in record['revised_ent_id']:
                    covered_target_ent_ids.add(record['target_ent_id'])
            covered_faithful_target_ent_ids = covered_target_ent_ids.intersection(set(faithful_orig_target_ids))
            ent_rel_frac = 1.0 if original_num_faithful_ents == 0 else len(covered_faithful_target_ent_ids) / original_num_faithful_ents
            for ent in noise_ents_sent:
                start, end = ent['BeginOffset'], ent['EndOffset']
                prev_chunk = revised_ents['text'][prev_start:start]
                prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
                annotated += prev_chunk.rstrip()
                category = ent['Category']
                type = ent['Type']
                ent_id = ent['ent_id']
                global_halluc = 0 if ent['ent_id'] in source_revised_overlap_ent_ids else 1
                local_halluc = 0 if ent['ent_id'] in target_revised_overlap_ent_ids else 1
                ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
                annotated += f'<e local={local_halluc} global={global_halluc} cat={category} type={type} id={ent_id}>' \
                             f'{ent_space_pre}' + revised_ents['text'][start:end] + '</e>'
                prev_start = end
                num_global += global_halluc
                num_local += local_halluc
            annotated += revised_ents['text'][prev_start:]
            annotations[f'{sent_idx}_{revise_idx}'] = annotated
            global_halluc_frac = 0 if len(noise_ents_sent) == 0 else num_global / len(noise_ents_sent)
            ent_info[f'{sent_idx}_{revise_idx}'] = {
                'num_ents': len(noise_ents_sent),
                'global_halluc_frac': global_halluc_frac,
                'ent_rel_frac': ent_rel_frac,
                'local_halluc': num_local,
                'global_halluc': num_global,
            }
        orig_annotated = ''
        prev_start = 0
        num_global = 0
        for ent in original_target_sent_ents:
            start, end = ent['BeginOffset'], ent['EndOffset']
            prev_chunk = original_text[prev_start:start]
            prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
            orig_annotated += prev_chunk.rstrip()
            category = ent['Category']
            type = ent['Type']
            ent_id = ent['ent_id']
            global_halluc = 0 if ent['ent_id'] in reg_overlap_ent_ids else 1
            ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
            orig_annotated += f'<e global={global_halluc} cat={category} type={type} id={ent_id}>' \
                         f'{ent_space_pre}' + original_text[start:end] + '</e>'
            prev_start = end
            num_global += global_halluc
        orig_annotated += original_text[prev_start:]
        annotations[f'{sent_idx}'] = orig_annotated
        ent_info[f'{sent_idx}'] = {
            'num_ents_original': len(original_target_sent_ents),
            'global_halluc_original': num_global
        }

    annotated_df = []
    for record in revise_sents.to_dict('records'):
        sent_idx = record['target_sent_idx']
        revise_idx = record['source_extract_code']
        try:
            record['text_original_annotated'] = annotations[str(sent_idx)]
            record['text_annotated'] = annotations[f'{sent_idx}_{revise_idx}']
        except:
            print(record['example_id'])
            example = record['example_id']
            raise Exception(f'Missing entity for {example}')
        record.update(ent_info[f'{sent_idx}'])
        record.update(ent_info[f'{sent_idx}_{revise_idx}'])
        annotated_df.append(record)
    annotated_df = pd.DataFrame(annotated_df)
    return annotated_df


def process(example_id, data_dir, experiment):
    noise_fn = os.path.join(revise_dir, 'output', experiment, f'{example_id}.csv')
    noise_ent_fn = os.path.join(revise_dir, 'acm_output', experiment, f'{example_id}.json')
    noise_merge_fn = os.path.join(revise_dir, 'ent_merges', experiment, f'{example_id}.csv')

    regular_ent_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.json')
    regular_ent_merge_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.csv')
    noise_sents = pd.read_csv(noise_fn)
    try:
        noise_merges = pd.read_csv(noise_merge_fn)
    except:
        noise_merges = None
    with open(noise_ent_fn, 'r') as fd:
        noise_ents = ujson.load(fd)

    with open(regular_ent_fn, 'r') as fd:
        reg_ents = ujson.load(fd)
    try:
        reg_ent_merges = pd.read_csv(regular_ent_merge_fn)
    except:
        reg_ent_merges = None

    annotated_df = annotate(reg_ents, reg_ent_merges, noise_ents, noise_merges, noise_sents)
    annotate_out_fn = os.path.join(annotate_dir, f'{example_id}.csv')
    annotated_df.to_csv(annotate_out_fn, index=False)

    eval_outputs = []
    processed_sents = set()
    for record in annotated_df.to_dict('records'):
        sent_idx = record['target_sent_idx']
        revise_idx = record['source_extract_code']
        revise_key = f'{sent_idx}_{revise_idx}'
        sent_key = str(sent_idx)
        orig_global_halluc_frac = record['global_halluc_original'] / max(1, record['num_ents_original'])

        if sent_key not in processed_sents:
            original_eval_row = {
                'example_id': example_id,
                'key': sent_key,
                'target_sent_idx': record['target_sent_idx'],
                'revise_idx': None,
                'version': 'original',
                'num_ents': record['num_ents_original'],
                'global_halluc': record['global_halluc_original'],
                'global_halluc_frac': orig_global_halluc_frac,
            }
            eval_outputs.append(original_eval_row)
            processed_sents.add(sent_key)

        revise_eval_row = {
            'example_id': example_id,
            'key': revise_key,
            'target_sent_idx': record['target_sent_idx'],
            'revise_idx': record['source_extract_code'],
            'version': 'revised',
            'num_ents': record['num_ents'],
            'local_halluc': record['local_halluc'],
            'global_halluc': record['global_halluc'],
            'global_halluc_frac': record['global_halluc_frac'],
            'ent_rel_frac': record['ent_rel_frac']
        }

        eval_outputs.append(revise_eval_row)

    eval_outputs = pd.DataFrame(eval_outputs)
    eval_out_fn = os.path.join(eval_dir, f'{example_id}.csv')
    eval_outputs.to_csv(eval_out_fn, index=False)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Evaluate revised outputs for entity overlap and annotate entities (ent type, hallucinated, etc.)')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default=None, required=True)
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-only_new', default=False, action='store_true')

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''
    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)

    revise_dir = os.path.join(data_dir, 'revise')
    revise_fns = glob(os.path.join(revise_dir, 'output', args.experiment, '*.csv'))
    ex_ids = set([x.split('/')[-1].replace('.csv', '') for x in revise_fns])

    revise_ent_fns = glob(os.path.join(revise_dir, 'acm_output', args.experiment, '*.json'))
    ent_ex_ids = set([x.split('/')[-1].replace('.json', '') for x in revise_ent_fns])

    example_ids = list(ex_ids.intersection(ent_ex_ids))
    if len(example_ids) > len(ex_ids):
        print(f'Found {len(ex_ids)} examples yet it shrinks to {len(example_ids)} when merging with entity info.')

    annotate_dir = os.path.join(revise_dir, 'annotated', args.experiment)
    os.makedirs(annotate_dir, exist_ok=True)

    eval_dir = os.path.join(revise_dir, 'ent_eval', args.experiment)
    os.makedirs(eval_dir, exist_ok=True)

    if args.only_new:
        json_pattern = eval_dir + '/*.csv'
        ent_fns = glob(json_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in ent_fns])
        print(f'Choosing not to re-evaluate entity overlaps for the {len(done_example_ids)} already completed examples')
        example_ids = [ex for ex in example_ids if ex not in done_example_ids]

    if args.cpu_frac == -1:
        statuses = list(tqdm(map(lambda ex_id: process(ex_id, data_dir, args.experiment),
                                 example_ids),
                             total=len(example_ids)))
    else:
        statuses = list(p_uimap(lambda ex_id: process(ex_id, data_dir, args.experiment),
                                example_ids))
    print(f'Successfully processed {sum(statuses)}/{len(statuses)} examples')
