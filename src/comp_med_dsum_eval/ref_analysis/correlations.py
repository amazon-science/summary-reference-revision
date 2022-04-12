# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import seaborn as sns
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots()
    sns.heatmap(num_df.corr(method='pearson'), annot=True, fmt='.1f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    plt.savefig('images/correlation.png', bbox_inches='tight', pad_inches=0)
