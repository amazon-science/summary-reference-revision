# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Displaying per-example statistics')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])

    args = parser.parse_args()
    data_dir = os.path.join(args.input_dir, args.target)
    in_fn = os.path.join(data_dir, 'stats.csv')
    df = pd.read_csv(in_fn)
    num_df = df.select_dtypes(include='number')
    corr_table = num_df.corr(method='pearson')

    correlations = np.array(corr_table['halluc_rate'].tolist())
    vars = corr_table.halluc_rate.axes[0].tolist()

    top_idxs = (-np.abs(correlations)).argsort()
    for idx in top_idxs:
        print(vars[idx], correlations[idx])
