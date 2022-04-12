# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing sentence index for high quality reference sentences.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--eval_size', default=1000, type=int)

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)
    input_dir = os.path.join(data_dir, 'revise')
    sent_fn = os.path.join(input_dir, f'sents_w_context.csv')
    print(f'Reading in reference sentences from {sent_fn}')
    sent_df = pd.read_csv(sent_fn)
    print(f'{len(sent_df)} sentences from summaries')
    sent_df = sent_df[sent_df['high_quality_w_ent']]
    print(f'{len(sent_df)} high quality sentences with >0 entities from summaries')
    test_example_ids = set(pd.read_csv(os.path.join(data_dir, 'test_example_ids.csv'))['example_id'])
    sent_df = sent_df[~sent_df['example_id'].isin(test_example_ids)]
    print(f'{len(sent_df)} sentences excluding test sentences')

    sent_df_sample = sent_df.sample(n=args.eval_size, replace=False, random_state=1992)[['example_id', 'sent_idx']]
    os.path.join(data_dir, 'test_example_ids.csv')
    out_test_fn = os.path.join(args.input_dir, args.target, 'high_quality', 'eval_examples.csv')
    sent_df_sample = sent_df_sample.assign(
        uid=sent_df_sample['example_id'] + '.' + sent_df_sample['sent_idx'].astype(str)
    )
    print(f'Saving {len(sent_df_sample)} sentence ids to {out_test_fn}')
    sent_df_sample.to_csv(out_test_fn, index=False)
