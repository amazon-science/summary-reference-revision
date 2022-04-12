# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import regex as re

import argparse
import pandas as pd
from p_tqdm import p_uimap

from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent


def remove_punc_whitespace(sent):
    return re.sub(r'\W+', '', remove_tags_from_sent(sent)).lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dont train perturber on any data from the test set discharge summaries '
                                     'or reviser eval set.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.cpu_frac = -1
    mini_str = '_mini' if args.debug else ''

    data_dir = os.path.join(args.input_dir, args.target)
    input_dir = os.path.join(data_dir, 'revise')
    sent_fn = os.path.join(input_dir, f'sents_w_context{mini_str}.csv')
    print(f'Reading in reference sentences from {sent_fn}')
    sent_df = pd.read_csv(sent_fn)
    test_example_ids = set(pd.read_csv(os.path.join(data_dir, 'test_example_ids.csv'))['example_id'])
    remove_df = sent_df[(sent_df['high_quality_w_ent']) | (sent_df['example_id'].isin(test_example_ids))]
    sents = list(set(list(map(remove_tags_from_sent, remove_df['target'].tolist()))))

    if args.cpu_frac == -1:
        outputs = list(set(list(map(remove_punc_whitespace, sents))))
    else:
        outputs = list(set(list(p_uimap(remove_punc_whitespace, sents, num_cpus=args.cpu_frac))))
    remove_sent_df = pd.DataFrame({'sent': outputs})
    out_fn = os.path.join(args.input_dir, 'dsum', 'remove_sents.csv')
    print(f'Saving {len(remove_sent_df)} unique sentences to remove to {out_fn}')
    remove_sent_df.to_csv(out_fn, index=False)
