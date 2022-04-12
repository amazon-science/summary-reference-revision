# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import itertools
import os
import ujson

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from p_tqdm import p_uimap
from tqdm import tqdm


def process(example_id, revise_dir, experiment):
    eval_dir = os.path.join(revise_dir, 'eval', experiment)
    gen_dir = os.path.join(revise_dir, 'output', experiment)
    ent_eval_dir = os.path.join(revise_dir, 'ent_eval', experiment)

    gen_df = pd.read_csv(os.path.join(gen_dir, f'{example_id}.csv'))
    ent_df = pd.read_csv(os.path.join(ent_eval_dir, f'{example_id}.csv'))
    # eval_df = pd.read_csv(os.path.join(eval_dir, f'{example_id}.csv'))
    key2halluc = dict(zip(ent_df['key'], ent_df['global_halluc_frac']))
    key2entrelfrac = dict(zip(ent_df['key'], ent_df['ent_rel_frac']))

    records = gen_df.to_dict('records')
    heat_map = []
    density_tradeoff = []
    extract_cov_tradeoff = []
    aug_records = []
    for record in records:
        target_sent_idx = record['target_sent_idx']
        revise_idx = record['source_extract_code']
        key = f'{target_sent_idx}_{revise_idx}'
        halluc_frac = key2halluc[key]
        ent_rel_frac = key2entrelfrac[key]
        record['global_halluc_frac'] = halluc_frac
        record['ent_rel_frac'] = ent_rel_frac
        heat_map.append({
            'source_extract_code': record['source_extract_code'],
            'input_extract_code': record['input_extract_code'],
            'gen_input_sim': record['gen_input_sim'],
            'gen_context_sim_improve': record['gen_context_sim_improve'],
        })

        extract_cov_tradeoff.append({
            'source_coverage': record['source_coverage'],
            'gen_input_sim': record['gen_input_sim'],
            'gen_context_sim_improve': record['gen_context_sim_improve'],
            'halluc_frac': halluc_frac,
            'ent_rel_frac': ent_rel_frac
        })

        density_tradeoff.append({
            'source_density': record['source_density'],
            'gen_input_sim': record['gen_input_sim'],
            'gen_context_sim_improve': record['gen_context_sim_improve'],
            'halluc_frac': halluc_frac,
            'ent_rel_frac': ent_rel_frac
        })

        aug_records.append(record)

    return density_tradeoff, extract_cov_tradeoff, heat_map, aug_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compile the results of the evaluations')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default='yay_repeat')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-overleaf', default=False, action='store_true')

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    data_dir = os.path.join(args.input_dir, args.target)
    revise_dir = os.path.join(data_dir, 'revise')
    gen_dir = os.path.join(revise_dir, 'output', args.experiment)
    results_dir = os.path.join(revise_dir, 'results', args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    assert os.path.exists(data_dir)
    fns = glob(gen_dir + '/*.csv')
    ex_ids = [fn.split('/')[-1].replace('.csv', '') for fn in fns]
    if args.debug:
        args.cpu_frac = -1

    if args.cpu_frac == -1:
        outputs = list(tqdm(map(lambda ex_id: process(ex_id, revise_dir, args.experiment), ex_ids), total=len(ex_ids)))
    else:
        outputs = list(
            p_uimap(lambda ex_id: process(ex_id, revise_dir, args.experiment), ex_ids, num_cpus=args.cpu_frac))

    density_tradeoff = pd.DataFrame(list(itertools.chain(*[x[0] for x in outputs])))
    extract_cov_tradeoff = pd.DataFrame(list(itertools.chain(*[x[1] for x in outputs])))
    heat_map = pd.DataFrame(list(itertools.chain(*[x[2] for x in outputs])))
    stats_df = pd.DataFrame(list(itertools.chain(*[x[3] for x in outputs])))

    table_outputs = []
    cols = ['global_halluc_frac', 'gen_context_cov_improve', 'ent_rel_frac']
    for col in cols:
        for source_extract_code in range(5, 10):
            s = stats_df[stats_df['source_extract_code'] == source_extract_code]
            table_val = str(round(s[col].dropna().mean() * 100, 1))
            table_outputs.append(table_val)
    print(f'Table string for experiment {args.experiment}')
    print(' & '.join(table_outputs))

    # Corel columns
    # source_extract_code
    # input_extract_code
    # global_halluc_frac
    # ent_rel_frac
    # gen_context_sim_improve
    # gen_input_sim
    corel_cols = ['source_extract_code', 'input_extract_code', 'global_halluc_frac', 'ent_rel_frac',
                  'gen_context_sim_improve', 'gen_input_sim', 'source_coverage', 'source_density']
    num_df = stats_df[corel_cols]
    fig, ax = plt.subplots()
    sns.heatmap(num_df.corr(method='pearson'), annot=True, fmt='.1f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    correl_fn = os.path.join(results_dir, 'correlation.png')
    print(f'Saving correlation matrix heatmap {correl_fn}')
    plt.savefig(correl_fn, bbox_inches='tight', pad_inches=0)
    plt.clf()

    # grouped = tuple(heat_map.groupby(by=['source_extract_code', 'input_extract_code']))
    #
    # # heat_map_data = []
    # rows = heat_map['input_extract_code'].max() + 1
    # cols = heat_map['source_extract_code'].max() + 1
    # context_heatmap = np.zeros([rows, cols])
    # input_heatmap = np.zeros([rows, cols])
    # for (source_extract_code, input_extract_code), subdf in grouped:
    #     context_heatmap[min(input_extract_code, 6), source_extract_code] = subdf['gen_context_sim_improve'].mean()
    #     input_heatmap[min(input_extract_code, 6), source_extract_code] = subdf['gen_context_sim_improve'].mean()
    # # heat_map_data = pd.DataFrame(heat_map_data)
    # # con_min, con_max = heat_map_data['gen_context_sim_improve'].min(), heat_map_data['gen_context_sim_improve'].max()
    #
    # d_fn = os.path.join(results_dir, 'density_tradeoff.csv')
    # e_fn = os.path.join(results_dir, 'extract_cov_tradeoff.csv')
    # h_fn = os.path.join(results_dir, 'heat_map.csv')
    #
    # print(f'Saving density tradeoff to {d_fn}')
    # density_tradeoff.to_csv(d_fn, index=False)
    # print(f'Saving extractive coverage tradeoff to {e_fn}')
    # extract_cov_tradeoff.to_csv(e_fn, index=False)
    # print(f'Saving control code heatmap data {h_fn}')
    # # heat_map_data.to_csv(h_fn, index=False)
    # #
    # # num_df = stats_df.select_dtypes('number')
    # # fig, ax = plt.subplots()
    # # sns.heatmap(num_df.corr(method='pearson'), annot=True, fmt='.1f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    # # ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    # # correl_fn = os.path.join(results_dir, 'correlation.png')
    # # print(f'Saving correlation matrix heatmap {correl_fn}')
    # # plt.savefig(correl_fn, bbox_inches='tight', pad_inches=0)
    # # plt.clf()
