# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import defaultdict, Counter
import itertools
from glob import glob
import os
import ujson

import numpy as np
from tqdm import tqdm
from p_tqdm import p_uimap

from comp_med_dsum_eval.preprocess.entity.process_ents import filter_ents
from comp_med_dsum_eval.preprocess.entity.entity_utils import ENT_TYPE_MAP


def ent_stats(ent_arr):
    num_ents = [len(ents) for ents in ent_arr]
    avg_num = np.mean(num_ents)
    flat_ent_types = list(itertools.chain(*ent_arr))
    n = len(flat_ent_types)
    count_by_type = Counter(flat_ent_types)
    stats = {'num': avg_num}
    for ent_type, count in count_by_type.items():
        stats[ent_type] = count_by_type[stats['num']]
    pass


def process(fn):
    with open(fn, 'r') as fd:
        ents = ujson.load(fd)

    source_ents_flat = list(itertools.chain(*[filter_ents(ent_obj['ents']) for ent_obj in ents['source']]))
    target_ents_flat = list(itertools.chain(*[filter_ents(ent_obj['ents']) for ent_obj in ents['target']]))
    ents_flat = source_ents_flat + target_ents_flat
    resolved_source_types = [ENT_TYPE_MAP[x['Type']] for x in source_ents_flat]
    resolved_target_types = [ENT_TYPE_MAP[x['Type']] for x in target_ents_flat]
    return [
        {'text': x['Text'], 'type': x['Type'], 'category': x['Category']} for x in ents_flat
    ], resolved_source_types, resolved_target_types


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Storing Dictionary ')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, args.target)
    ent_dir = os.path.join(input_dir, 'acm_output')
    pattern = ent_dir + '/*.json'
    print('Searching for entities...')
    fns = glob(pattern)
    if args.cpu_frac == -1:
        outputs = list(tqdm(map(process, fns)))
    else:
        outputs = list(p_uimap(process, fns, num_cpus=args.cpu_frac))

    source_type_counts = {}
    target_type_counts = {}
    source_ents = [x[1] for x in outputs]
    target_ents = [x[2] for x in outputs]
    counts = {}

    flat_ents = list(itertools.chain(*[x[0] for x in outputs]))
    for ent in tqdm(flat_ents):
        if ent['type'] not in counts:
            counts[ent['type']] = defaultdict(int)
        counts[ent['type']][ent['text']] += 1

    output_json = {}
    for type, count_obj in counts.items():
        texts = list(count_obj.keys())
        v = np.array(list(count_obj.values()))
        p = list(v / v.sum())
        output_json[type] = {
            'text': texts,
            'p': p
        }

    out_fn = os.path.join(input_dir, 'ent_inventory.json')
    with open(out_fn, 'w') as fd:
        ujson.dump(output_json, fd)
