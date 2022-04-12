# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
np.random.seed(1992)  # reproducible
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Displaying per-example statistics')
    parser.add_argument('--input_dir', default='/efs/griadams/hpi/')
    parser.add_argument('--num_samples', default=10, type=int)

    args = parser.parse_args()

    stats_fn = os.path.join(args.input_dir, 'stats.csv')
    stats_df = pd.read_csv(stats_fn)
    stat_cols = list(stats_df.columns)
    summary_fn = os.path.join(args.input_dir, 'summary_dataset_ent.csv')
    summary_df = pd.read_csv(summary_fn)

    merged_df = stats_df.merge(summary_df, on='example_id')

    ranges = {
        'low': (0, round(len(stats_df) * 0.1)),
        'mid': (round(len(stats_df) * 0.45), round(len(stats_df) * 0.55)),
        'high': (round(len(stats_df) * 0.9), len(stats_df))
    }

    vars_of_interest = ['compression', 'density', 'coverage',
                        'ent_coverage', 'ent_recall', 'target_toks', 'target_sents']

    for var in vars_of_interest:
        print(f'Writing out examples for {var}')
        example_order = merged_df.sort_values(by=var).reset_index(drop=True)
        var_dir = os.path.join('buckets', var)
        for section, (start, end) in ranges.items():
            var_df = example_order[start:end].sample(n=args.num_samples, replace=False)
            sec_dir = os.path.join(var_dir, section)
            os.makedirs(sec_dir, exist_ok=True)
            var_output = []
            for record in var_df.to_dict('records'):
                example_id = record['example_id']
                out_fn = os.path.join(sec_dir, f'{example_id}.txt')

                out_str = f'This example is a {section} example with respect to {var}\n\n'
                for col in stat_cols:
                    out_str += f'{col}={record[col]}\n'
                out_str += '\n'

                out_str += ('-' * 50) + 'START OF REFERENCE' + ('-' * 50) + '\n'
                out_str += ' '.join(record['target'].split('<SEP>')) + '\n'
                out_str += ('-' * 50) + 'END OF REFERENCE' + ('-' * 50) + '\n\n'

                out_str += ('-' * 50) + 'START OF SOURCE' + ('-' * 50) + '\n'
                out_str += ' '.join(record['source'].split('<SEP>')) + '\n'
                out_str += ('-' * 50) + 'END OF SOURCE' + ('-' * 50) + '\n'

                with open(out_fn, 'w') as fd:
                    fd.write(out_str)
