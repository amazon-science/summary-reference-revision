# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from glob import glob
import ujson
import os

import argparse
import pandas as pd
import numpy as np
from p_tqdm import p_uimap
from tqdm import tqdm

from comp_med_dsum_eval.ref_reviser.alignments.utils import keep_sent


def _process(example):
    example_id = example['example_id']
    src = example['source']
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
        'source': {
            'sent_idxs': source_sent_idxs,
            'sents': source_sents,
            'improvements': source_improvements
        },
    }
    row.update(example['stats'])
    return row


def process(fn, quality_uids, filter_out_embed_info=False):
    with open(fn, 'r') as fd:
        examples = ujson.load(fd)
        quality_examples = []
        for example in examples:
            uid = example['example_id'] + '.' + str(example['target']['sent_idx'])
            if uid in quality_uids:
                if filter_out_embed_info:
                    example = _process(example)
                quality_examples.append(example)
        return quality_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collecting high/low quality examples for ref reviser dataset.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--quality', default='high', choices=['high', 'low'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)
    revise_dir = os.path.join(data_dir, 'revise', 'contexts')

    quality_fn = os.path.join(data_dir, 'revise', 'sent_quality.csv')
    quality_df = pd.read_csv(quality_fn)
    if args.quality == 'high':
        quality_df = quality_df[quality_df['high_quality_w_ent']]
        filter_out_embed_info = False
    elif args.quality == 'low':
        quality_df = quality_df[~quality_df['high_quality']]
        filter_out_embed_info = True
    else:
        raise Exception(f'Unrecognized quality cohort -> {args.quality}')
    print(f'Found {len(quality_df)} {args.quality} quality sentences')
    quality_uids = set(quality_df['example_id'] + '.' + quality_df['sent_idx'].astype(str))

    fns = glob(revise_dir + '/*.json')
    if args.cpu_frac == -1:
        outputs = list(itertools.chain(*list(tqdm(map(lambda fn: process(
            fn, quality_uids, filter_out_embed_info=filter_out_embed_info), fns), total=len(fns)))))
    else:
        outputs = list(itertools.chain(*list(p_uimap(lambda fn: process(
            fn, quality_uids, filter_out_embed_info=filter_out_embed_info), fns, num_cpus=args.cpu_frac))))

    out_fn = os.path.join(data_dir, 'revise', f'{args.quality}_quality_w_context.json')
    print(f'Saving {len(outputs)} sentence-level examples to {out_fn}')
    with open(out_fn, 'w') as fd:
        ujson.dump(outputs, fd)

    # We need a smaller evaluation cohort for the low quality examples
    if args.quality == 'low':
        test_example_ids = set(pd.read_csv(os.path.join(data_dir, 'test_example_ids.csv'))['example_id'])
        out_no_test = [x for x in outputs if x['example_id'] not in test_example_ids]
        np.random.seed(1992)
        eval_outs = list(np.random.choice(outputs, size=(1000,), replace=False))
        eval_out_fn = os.path.join(data_dir, 'revise', f'{args.quality}_quality_w_context_eval.json')
        print(f'Saving {len(eval_outs)} sentence-level examples to {eval_out_fn}')
        with open(eval_out_fn, 'w') as fd:
            ujson.dump(eval_outs, fd)

    mini_out_fn = os.path.join(data_dir, 'revise', f'{args.quality}_quality_w_context_mini.json')
    mini_outputs = list(np.random.choice(outputs, size=(128, ), replace=False))
    print(f'Saving {len(mini_outputs)} sentence-level examples to {mini_out_fn}')
    with open(mini_out_fn, 'w') as fd:
        ujson.dump(mini_outputs, fd)
