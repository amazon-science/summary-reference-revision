# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Render perturber results in paper table format')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default=None, required=True)
    parser.add_argument('-overleaf', default=False, action='store_true')

    args = parser.parse_args()
    data_dir = os.path.join(args.input_dir, args.target)
    noise_dir = os.path.join(data_dir, 'perturb', args.experiment)
    df = pd.read_csv(os.path.join(noise_dir, 'results', 'results.csv'))
    df = df[df['version'] == 'perturbed']
    df['bert_bs_f1'] = (2 * df['bert_bs_precision'] * df['bert_bs_recall'] /
                        (df['bert_bs_precision'] + df['bert_bs_recall']))
    df['correctness'] = 1 - df['fake_frac']
    diversity_fn = os.path.join(noise_dir, 'diversity', 'extractive_frag.csv')
    diversity_df = pd.read_csv(diversity_fn)
    diversity_df['diversity'] = 1 - diversity_df['coverage']

    col_order = [
        (['local_halluc_frac', 'global_halluc_frac'], 'Hallucination Fraction (Local/Global)', df, 100, 0),
        ('bert_bs_f1', 'F1-BERTScore to Input', df, 100, 1),
        ('nsp', 'Coherence', df, 100, 1),
        ('correctness', 'Correctness', df, 100, 1),
        ('diversity', 'Diversity', diversity_df, 100, 0)
    ]

    means = []
    delim = ' & ' if args.overleaf else ' '
    for col, name, df, scale, dec_places in col_order:
        if type(col) == list:
            mean = '/'.join(list(map(lambda col: str(round(df[col].dropna().mean() * scale, dec_places)), col)))
        else:
            mean = str(round(df[col].dropna().mean() * scale, dec_places))
        print(f'{name}: {mean}')
        means.append(mean)
    print(delim.join(means))
    print(f'Done showing results for {args.experiment}')
