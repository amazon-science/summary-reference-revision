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


def _get_num_attr(text, attr):
    return None if attr not in text else int(re.search(rf'<{attr}-(\d+)>', text).group(1))


def extract_perturb_params(record):
    if 'masked_str' not in record:
        return {}
    return {
        'span_remove': _get_num_attr(record['masked_str'], 'span-remove'),
        'ent_add': _get_num_attr(record['masked_str'], 'ent-add'),
        'ent_remove': _get_num_attr(record['masked_str'], 'ent-remove'),
        'shuffle_orderliness': _get_num_attr(record['masked_str'], 'shuffle'),
    }


def annotate(reg_ents, reg_merge_scores, noise_ents, noise_merge_scores, noise_sents):
    target_ents = reg_ents['target']

    target_ents_flat = list(itertools.chain(
        *[filter_ents(add_ent_id(ent_obj, 'target')) for ent_obj in target_ents]
    ))

    if reg_merge_scores is None:
        reg_overlap_ent_ids = set()
    else:
        reg_merge_scores = reg_merge_scores[reg_merge_scores['should_merge']]
        reg_overlap_ent_ids = set(reg_merge_scores['target_ent_id'].tolist())

    if noise_merge_scores is None:
        source_perturbed_overlap_ent_ids = set()
        target_perturbed_overlap_ent_ids = set()
    else:
        noise_merge_scores = noise_merge_scores[noise_merge_scores['should_merge']]
        source_perturbed_scores = noise_merge_scores[noise_merge_scores['relation'] == 'source-perturbed']
        target_perturbed_scores = noise_merge_scores[noise_merge_scores['relation'] == 'target-perturbed']
        source_perturbed_overlap_ent_ids = set(source_perturbed_scores['perturbed_ent_id'].tolist())
        target_perturbed_overlap_ent_ids = set(target_perturbed_scores['perturbed_ent_id'].tolist())

    annotations = {}
    ent_info = {}
    for sent_idx in noise_ents:
        original_target_sent_ents = [
            x for x in target_ents_flat if x['sent_idx'] == int(sent_idx) and x['text_type'] == 'sent']
        original_text = [x['text'] for x in target_ents if x['dtype'] == 'sent' and x['sent_idx'] == int(sent_idx)]
        assert len(original_text) == 1
        original_text = original_text[0]
        perturbed_objs = noise_ents[sent_idx]['perturbed']
        for perturb_idx, perturbed_ents in perturbed_objs.items():
            annotated = ''
            noise_ents_sent = filter_ents(flatten_ents(perturbed_ents['ents']))
            prev_start = 0
            num_global, num_local = 0, 0
            for ent in noise_ents_sent:
                start, end = ent['BeginOffset'], ent['EndOffset']
                prev_chunk = perturbed_ents['text'][prev_start:start]
                prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
                annotated += prev_chunk.rstrip()
                category = ent['Category']
                type = ent['Type']
                ent_id = ent['ent_id']
                global_halluc = 0 if ent['ent_id'] in source_perturbed_overlap_ent_ids else 1
                local_halluc = 0 if ent['ent_id'] in target_perturbed_overlap_ent_ids else 1
                ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
                annotated += f'<e local={local_halluc} global={global_halluc} cat={category} type={type} id={ent_id}>' \
                             f'{ent_space_pre}' + perturbed_ents['text'][start:end] + '</e>'
                prev_start = end
                num_global += global_halluc
                num_local += local_halluc
            annotated += perturbed_ents['text'][prev_start:]
            annotations[f'{sent_idx}_{perturb_idx}'] = annotated
            ent_info[f'{sent_idx}_{perturb_idx}'] = {
                'num_ents': len(noise_ents_sent),
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
    for record in noise_sents.to_dict('records'):
        sent_idx = record['sent_idx']
        perturb_idx = record['perturb_idx']
        try:
            record['text_original_annotated'] = annotations[str(sent_idx)]
            record['text_annotated'] = annotations[f'{sent_idx}_{perturb_idx}']
        except:
            print(record['example_id'])
            example = record['example_id']
            raise Exception(f'Missing entity for {example}')
        record.update(ent_info[f'{sent_idx}'])
        record.update(ent_info[f'{sent_idx}_{perturb_idx}'])
        annotated_df.append(record)
    annotated_df = pd.DataFrame(annotated_df)
    return annotated_df


def process(example_id, noise_dir, data_dir, annotate_dir, eval_dir):
    noise_fn = os.path.join(noise_dir, 'output', f'{example_id}.csv')
    noise_ent_fn = os.path.join(noise_dir, 'acm_output', f'{example_id}.json')
    noise_merge_fn = os.path.join(noise_dir, 'ent_merges', f'{example_id}.csv')

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
        sent_idx = record['sent_idx']
        perturb_idx = record['perturb_idx']
        perturb_key = f'{sent_idx}_{perturb_idx}'
        sent_key = str(sent_idx)
        orig_global_halluc_frac = record['global_halluc_original'] / max(1, record['num_ents_original'])
        noise_global_halluc_frac = record['global_halluc'] / max(1, record['num_ents'])

        local_overlap = record['num_ents'] - record['local_halluc']
        recall = local_overlap / max(1, record['num_ents_original'])
        precision = local_overlap / max(1, record['num_ents'])
        f1 = 0 if recall + precision == 0 else (2 * recall * precision) / (precision + recall)

        if sent_key not in processed_sents:
            original_eval_row = {
                'example_id': example_id,
                'key': sent_key,
                'sent_idx': record['sent_idx'],
                'perturb_idx': None,
                'version': 'original',
                'num_ents': record['num_ents_original'],
                'global_halluc': record['global_halluc_original'],
                'global_halluc_frac': orig_global_halluc_frac,
            }
            eval_outputs.append(original_eval_row)
            processed_sents.add(sent_key)

        perturb_eval_row = {
            'example_id': example_id,
            'key': perturb_key,
            'sent_idx': record['sent_idx'],
            'perturb_idx': record['perturb_idx'],
            'version': 'perturbed',
            'num_ents': record['num_ents'],
            'local_halluc': record['local_halluc'],
            'global_halluc': record['global_halluc'],
            'local_halluc_frac': record['local_halluc'] / max(1, record['num_ents']),
            'global_halluc_frac': noise_global_halluc_frac,
            'local_ent_recall': recall,
            'local_ent_precision': precision,
            'local_ent_f1': f1,
            'diff_global_halluc_frac': noise_global_halluc_frac - orig_global_halluc_frac
        }

        perturb_eval_row.update(extract_perturb_params(record))
        eval_outputs.append(perturb_eval_row)

    eval_outputs = pd.DataFrame(eval_outputs)
    eval_out_fn = os.path.join(eval_dir, f'{example_id}.csv')
    eval_outputs.to_csv(eval_out_fn, index=False)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate perturb outputs for plausibility and semantic variance')
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

    noise_dir = os.path.join(data_dir, 'perturb', args.experiment)
    noise_fns = glob(noise_dir + '/output/*.csv')
    ex_ids = set([x.split('/')[-1].replace('.csv', '') for x in noise_fns])

    noise_ent_fns = glob(noise_dir + '/acm_output/*.json')
    ent_ex_ids = set([x.split('/')[-1].replace('.json', '') for x in noise_ent_fns])

    example_ids = list(ex_ids.intersection(ent_ex_ids))
    if len(example_ids) > len(ex_ids):
        print(f'Found {len(ex_ids)} examples yet it shrinks to {len(example_ids)} when merging with entity info.')

    annotate_dir = os.path.join(noise_dir, 'annotated')
    os.makedirs(annotate_dir, exist_ok=True)

    eval_dir = os.path.join(noise_dir, 'ent_eval')
    os.makedirs(eval_dir, exist_ok=True)

    if args.only_new:
        json_pattern = eval_dir + '/*.csv'
        ent_fns = glob(json_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in ent_fns])
        print(f'Choosing not to re-evaluate entity overlaps for the {len(done_example_ids)} already completed examples')
        example_ids = [ex for ex in example_ids if ex not in done_example_ids]

    if args.cpu_frac == -1:
        statuses = list(tqdm(map(lambda ex_id: process(ex_id, noise_dir, data_dir, annotate_dir, eval_dir),
                                 example_ids),
                             total=len(example_ids)))
    else:
        statuses = list(p_uimap(lambda ex_id: process(ex_id, noise_dir, data_dir, annotate_dir, eval_dir),
                                example_ids))
    print(f'Successfully processed {sum(statuses)}/{len(statuses)} examples')
