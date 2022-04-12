# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Displaying per-example statistics')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])

    args = parser.parse_args()
    data_dir = os.path.join(args.input_dir, args.target)
    in_fn = os.path.join(data_dir, 'stats.csv')
    stat_df = pd.read_csv(in_fn)
    stat_df = stat_df.assign(visit_id=stat_df['example_id'].apply(lambda x: x.split('_')[1]))
    visit_id_to_halluc = dict(zip(stat_df['visit_id'], stat_df['halluc_rate']))
    icu_df = pd.read_csv(os.path.join(args.input_dir, 'patients_time_spent_outside_icu.csv'))

    icu_df = icu_df.assign(halluc_rate=icu_df['hadm_id'].apply(lambda visit_id: visit_id_to_halluc.get(str(visit_id), None)))
    icu_df.dropna(subset=['halluc_rate', 'total_outside_icu'], inplace=True)
    icu_df.to_csv('icu_halluc.csv', index=False)

    print(pearsonr(icu_df['total_outside_icu'], icu_df['halluc_rate'])[0])
