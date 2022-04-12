# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Displaying per-example statistics')
    parser.add_argument('--input_dir', default='/efs/griadams')

    args = parser.parse_args()
    fn = os.path.join(args.input_dir, 'hpi', 'summary_dataset_ent_validation.csv')

    examples = pd.read_csv(fn)
    stat_df = pd.read_csv('/efs/griadams/hpi/stats.csv')
    examples = examples.merge(stat_df, on='example_id')
    viable_visits = set(pd.read_csv('/efs/griadams/viable_visits.csv')['HADM_ID'].apply(
        lambda x: str(x).split('.')[0]))
    examples = examples.assign(
        hadm_id=examples['example_id'].apply(lambda x: x.split('_')[-1]),
        source_hpi=examples['source'].apply(lambda x: 'history_present_illness' in x)
    )
    examples = examples.assign(
        has_admission=examples['hadm_id'].apply(lambda hadm_id: hadm_id in viable_visits)
    )

    admit_examples = examples[examples['has_admission']]

    n = len(examples)
    admit_n = len(admit_examples)

    source_hpi = examples['source_hpi'].sum()
    source_hpi_admit_only = admit_examples['source_hpi'].sum()

    print(f'Overall source HPI frequency: {source_hpi / n} ({source_hpi}/{n})')
    print(f'Source HPI frequency when there is an admission note: {source_hpi_admit_only / admit_n} ({source_hpi_admit_only}/{admit_n})')

    stat_cols = ['coverage', 'density', 'compression', 'ent_coverage', 'ent_recall', 'source_toks', 'target_toks']
    for col in stat_cols:
        print(f'{col}: Full={examples[col].dropna().mean()}, Admit Only={admit_examples[col].dropna().mean()}')
