# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import os
from glob import glob
import ujson
import regex as re

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm

from comp_med_dsum_eval.preprocess.constants import HTML_REGEX_NO_SPACE
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import get_attr
from comp_med_dsum_eval.perturber.utils import filter_df_for_eval, set_seeds
from comp_med_dsum_eval.preprocess.entity.entity_utils import ENT_ORDER as ENTITY_TYPES, ENT_TYPE_MAP

ENT_SWAP_FRAC = 0.5


def swap(text, record, ent_inventory, strategy):
    num_ents = len(re.findall(r'<e', text))
    num_ents_to_swap = round(np.random.beta(1, 1 / ENT_SWAP_FRAC - 1) * num_ents)
    ent_idxs_to_swap = set(list(np.random.choice(np.arange(num_ents), size=(num_ents_to_swap,), replace=False)))
    dig = min(num_ents_to_swap, 10)
    masked_str = f'<ent-add-{dig}> <ent-remove-{dig}>'

    related_inventory = defaultdict(list)
    for entity_type in ENTITY_TYPES:
        if type(record[entity_type]) != float and len(record[entity_type]) > 0:
            related_inventory[entity_type] = record[entity_type].split('<SEP>')

    tps = re.split(HTML_REGEX_NO_SPACE, text)
    removed_ents = []
    added_ents = []
    text_perturbed = ''
    ent_idx = -1
    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<e') or tp == '</e>':
            continue
        elif tp_idx > 0 and tps[tp_idx - 1].startswith('<e'):
            ent_idx += 1
            if ent_idx in ent_idxs_to_swap:
                ent_type = get_attr(tps[tp_idx - 1], 'type')
                ent_type_resolved = ENT_TYPE_MAP[ent_type]
                if strategy == 'related' and len(related_inventory[ent_type_resolved]) > 0:
                    sampled_ent = np.random.choice(related_inventory[ent_type_resolved], size=(1, ))[0]
                else:
                    sampled_ent = np.random.choice(
                        ent_inventory[ent_type]['text'], p=ent_inventory[ent_type]['p'], size=(1, ))[0]
                removed_ents.append(tp.strip())
                added_ents.append(sampled_ent)
                lead_space = re.search(r'^([\s]+)', tp)
                trail_space = re.search(r'([\s]+)$', tp)
                lead_space_str = '' if lead_space is None else lead_space.group(0)
                trail_space_str = '' if trail_space is None else trail_space.group(0)
                text_perturbed += lead_space_str + sampled_ent + trail_space_str
            else:
                text_perturbed += tp
        else:
            text_perturbed += tp
    return text_perturbed, masked_str, removed_ents, added_ents


def generate_swaps(example_id, records, noise_dir, ent_inventory, strategy, samples=5, verbose=False):
    outputs = []
    for record in records:
        assert example_id == record['example_id']
        text = record['text']
        for perturb_idx in range(samples):
            text_perturbed, masked_str, removed_ents, added_ents = swap(text, record, ent_inventory, strategy)
            if verbose:
                print(text, ' -> ', text_perturbed)
            outputs.append({
                'example_id': example_id,
                'sent_idx': record['sent_idx'],
                'perturb_idx': perturb_idx,
                'text': text_perturbed,
                'text_original': text,
                'masked_str': masked_str,
                'added_ents': '<e>'.join(added_ents),
                'removed_ents': '<e>'.join(removed_ents)
            })
    outputs = pd.DataFrame(outputs)
    out_fn = os.path.join(noise_dir, f'{example_id}.csv')
    outputs.to_csv(out_fn, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating entity swaps')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default='swap')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--strategy', default='random', choices=['random', 'related'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-only_eval', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('-only_new', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.cpu_frac = -1

    # Set same random seed for each run
    set_seeds(args.seed)

    data_dir = os.path.join(args.input_dir, args.target)
    noise_dir = os.path.join(data_dir, 'perturb', args.experiment + '_' + args.strategy, 'output')
    print(f'Creating {noise_dir} directory if it doesn\'t already exist')
    os.makedirs(noise_dir, exist_ok=True)

    mini_str = '_mini' if args.debug else ''
    data_fn = os.path.join(data_dir, 'high_quality', f'sents_w_related_ents{mini_str}.csv')
    print(f'Reading in data from {data_fn} and dropping empty rows...')
    data_df = pd.read_csv(data_fn)
    if args.only_eval:
        data_df = filter_df_for_eval(data_df, data_dir)
        print(f'Filtered down to {len(data_df)} eval examples')

    if args.only_new:
        csv_pattern = noise_dir + '/*.csv'
        ent_fns = glob(csv_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in ent_fns])
        print(f'Choosing not to re-evaluate entity overlaps for the {len(done_example_ids)} already completed examples')
        data_df = data_df[~data_df['example_id'].isin(done_example_ids)]

    example_records = list(data_df.groupby('example_id'))

    ent_inventory_fn = os.path.join(data_dir, 'ent_inventory.json')
    with open(ent_inventory_fn, 'r') as fd:
        ent_inventory = ujson.load(fd)

    if args.cpu_frac == -1:
        list(tqdm(map(
            lambda x: generate_swaps(
                x[0], x[1].to_dict('records'), noise_dir, ent_inventory, args.strategy, verbose=args.debug),
            example_records
        ), total=len(example_records)))
    else:
        list(p_uimap(
            lambda x: generate_swaps(
                x[0], x[1].to_dict('records'), noise_dir, ent_inventory, args.strategy, verbose=args.debug),
            example_records,
            num_cpus=args.cpu_frac
        ))
