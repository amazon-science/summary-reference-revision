# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('-overleaf', default=False, action='store_true')
    parser.add_argument('--experiment', default=None, required=True)  # e.g., no_ent_admit

    args = parser.parse_args()

    print(f'Showing results for {args.experiment}')
    output_df = pd.read_csv(f'/efs/griadams/bhc/mimic_sum/results/{args.experiment}/outputs.csv')
    # Annotated with entities and includes entity overlap.  Generated from ent_evaluation.py
    ent_df = pd.read_csv(f'/efs/griadams/bhc/mimic_sum/results/{args.experiment}/outputs_annotated.csv')
    ent_df['halluc_frac'] = ent_df['pred_global_halluc'].div(ent_df['pred_num_ents']).replace(np.inf, 0)
    output_df['correctness'] = 1 - output_df['pred_fake_frac']

    col_order = [
        # ('pred_num_ents', 'Number of Predicted Entities', ent_df, 1),
        ('halluc_frac', 'Hallucination Rate', ent_df, 100),
        ('bert_bs_src_cov', 'BERTScore Source Precision', output_df, 100),  # this is a messup in my naming.
        ('bert_bs_src_prec', 'BERTScore Source Recall', output_df, 100),  # this is a messup in my naming.
        ('bert_bs_src_f1', 'BERTScore Source F1', output_df, 100),
        ('faithful_ent_recall', 'Faithfulness-Adjusted Coverage', ent_df, 100),
        ('pred_nsp', 'Coherence', output_df, 100),
        ('correctness', 'Correctness', output_df, 100),
        # ('bert_bs_cov', 'BERTScore Noisy Reference Precision', output_df, 100),  # this is a messup in my naming.
        # ('bert_bs_prec', 'BERTScore Reference Recall', output_df, 100),  # this is a messup in my naming.
        # ('bert_bs_f1', 'BERTScore Reference F1', output_df, 100)
    ]

    means = []
    delim = ' & ' if args.overleaf else ' '
    for col, name, df, scale in col_order:
        mean = str(round(df[col].dropna().mean() * scale, 1))
        print(f'{name}: {mean}')
        means.append(mean)
    print(delim.join(means))
    print(f'Done showing results for {args.experiment}')
