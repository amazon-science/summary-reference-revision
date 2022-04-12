# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Displaying per-example statistics')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])

    args = parser.parse_args()
    data_dir = os.path.join(args.input_dir, args.target)

    in_fn = os.path.join(data_dir, 'train_stats.csv')
    out_dir = os.path.join('images', args.target)
    os.makedirs(out_dir, exist_ok=True)

    stats = pd.read_csv(in_fn)
    cols = [x for x in stats.columns if x not in {'example_id', 'fragments'}]
    for col in tqdm(cols, total=len(cols)):
        print(f'Building density plot for {col}')
        valid = stats[col].dropna()
        ax = valid.plot.kde(title=f'Distribution of {col} in MIMIC dataset')
        max_x = round(np.percentile(valid, 90)) + 1
        if 'coverage' in col or 'recall' in col:
            max_x = 1
        ax.set_xlim(0, max_x)
        plt.savefig(os.path.join(out_dir, '{col}.png'))
        plt.clf()
