# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import os

import argparse
from nltk import word_tokenize
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm

from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent, sents_from_html


def unique_example_toks(record, lower=True):
    source_sents = ' '.join([remove_tags_from_sent(x) for x in sents_from_html(record['source'])])
    target_sents = ' '.join([remove_tags_from_sent(x) for x in sents_from_html(record['target'])])
    full_str = source_sents + ' ' + target_sents
    if lower:
        full_str = full_str.lower()
    toks = list(set(list(filter(lambda x: len(x) > 0, word_tokenize(full_str)))))
    return toks


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing TF-IDF over corpus')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('-uncased', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.5, type=float)

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)
    in_fn = os.path.join(data_dir, 'summary_dataset.csv')
    print('Reading in {}'.format(in_fn))
    df = pd.read_csv(in_fn)
    records = df.to_dict('records')

    if args.uncased:
        print('Lowercasing all text.')

    toks = list(p_uimap(lambda record: unique_example_toks(record, lower=args.uncased), records))
    n = float(len(toks))
    target_cts = defaultdict(int)
    for tok_set in tqdm(toks):
        for tok in tok_set:
            target_cts[tok] += 1

    tf_df = pd.DataFrame([{'tok': tok, 'ct': ct, 'idf': float(np.log(n / ct))} for tok, ct in target_cts.items()])
    cased_str = '_uncased' if args.uncased else ''
    out_fn = os.path.join(data_dir, f'toks{cased_str}.csv')
    print(f'Saving {len(tf_df)} tokens to {out_fn}')
    tf_df.to_csv(out_fn, index=False)
