# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate NSP scores')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--splits', default='validation,train,test', choices=
    ['train', 'validation', 'test', 'train,validation', 'validation,test', 'train,test', 'train,validation,test'])

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)

    high_quality_examples = []
    low_quality_examples = []
    mid_quality_examples = []
    for split in args.splits.split(','):
        df = pd.read_csv(os.path.join(data_dir, f'{split}_stats.csv'))
        high_ex = df[(df['source_coverage'] > 0) & (df['coverage'] >= 0.75) & (df['halluc_rate'] <= 0.1)
                ]['example_id'].tolist()
        print(f'Adding {len(high_ex)} high quality examples from {split} set')
        high_quality_examples += high_ex

        remaining_df = df[~df['example_id'].isin(set(high_ex))]
        low_ex = remaining_df[
            (remaining_df['coverage'] < 0.5) | (remaining_df['halluc_rate'] > 0.5)]['example_id'].tolist()
        print(f'Adding {len(low_ex)} low quality examples from {split} set')
        low_quality_examples += low_ex

        mid_ex = remaining_df[~remaining_df['example_id'].isin(set(low_ex))]['example_id'].tolist()
        mid_quality_examples += mid_ex
        print(f'Adding {len(mid_ex)} mid quality examples from {split} set')

    for cohort, ex in [('low', low_quality_examples), ('mid', mid_quality_examples), ('high', high_quality_examples)]:
        out_fn = os.path.join(data_dir, f'{cohort}_coverage_examples.csv')
        print(f'Saving {len(ex)} to {out_fn}')
        pd.DataFrame({'example_id': ex}).to_csv(out_fn, index=False)
