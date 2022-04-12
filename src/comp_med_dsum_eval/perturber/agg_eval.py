# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import itertools
import os

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from p_tqdm import p_uimap


ORIG_ENT_COLS = {
    'num_ents',
    'global_halluc',
    'global_halluc_frac'
}


PERTURB_ENT_COLS = {
    'num_ents',
    'local_halluc',
    'global_halluc',
    'local_halluc_frac',
    'global_halluc_frac',
    'local_ent_recall',
    'local_ent_precision',
    'local_ent_f1',
    'diff_global_halluc_frac'
}


def process(fn, nsp_dir, ent_dir, uid_filter=None):
    df = pd.read_csv(fn)
    suffix = fn.split('/')[-1]
    try:
        nsp_df = pd.read_csv(os.path.join(nsp_dir, suffix))
        key2nsp = dict(zip(nsp_df['key'], nsp_df['nsp']))
    except:
        key2nsp = None

    try:
        ent_df = pd.read_csv(os.path.join(ent_dir, suffix))
        key2ent = dict(tuple(ent_df.groupby(by='key')))
    except:
        key2ent = None

    sent_info = df[df['version'] == 'original']
    perturb_info = df[df['version'] == 'perturbed']

    outputs = []
    sent_frac_map = dict(zip(sent_info['sent_idx'], sent_info['fake_frac']))
    sent_max_map = dict(zip(sent_info['sent_idx'], sent_info['fake_score_max']))
    for record in perturb_info.to_dict('records'):
        example_id = record['example_id']
        sent_idx = record['sent_idx']
        uid = f'{example_id}.{sent_idx}'
        if uid_filter is not None and uid not in uid_filter:
            continue
        fake_frac_delta = record['fake_frac'] - sent_frac_map[sent_idx]
        fake_max_delta = record['fake_score_max'] - sent_max_map[sent_idx]
        record['fake_frac_delta'] = fake_frac_delta
        record['fake_max_delta'] = fake_max_delta
        if key2nsp is not None:
            try:
                record['nsp'] = key2nsp[record['key']]
            except:
                k = record['key']
                print(f'Missing NSP key={k} for {fn}')
        if key2ent is not None:
            ent_metrics = key2ent[record['key']][PERTURB_ENT_COLS].to_dict('records')[0]
            record.update(ent_metrics)
        outputs.append(record)
    for record in sent_info.to_dict('records'):
        example_id = record['example_id']
        sent_idx = record['sent_idx']
        uid = f'{example_id}.{sent_idx}'
        if uid_filter is not None and uid not in uid_filter:
            continue
        if key2nsp is not None:
            try:
                record['nsp'] = key2nsp[record['key']]
            except:
                k = record['key']
                print(f'Missing NSP key={k} for {fn}')
        if key2ent is not None:
            ent_metrics = key2ent[record['key']][ORIG_ENT_COLS].to_dict('records')[0]
            record.update(ent_metrics)
        outputs.append(record)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compile the results of the evaluations')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default='ent_sample')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-only_eval', default=False, action='store_true')

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    data_dir = os.path.join(args.input_dir, args.target, 'perturb', args.experiment)
    assert os.path.exists(data_dir)
    noise_eval_dir = os.path.join(data_dir, 'eval')
    noise_nsp_dir = os.path.join(data_dir, 'nsp')
    noise_ent_eval_dir = os.path.join(data_dir, 'ent_eval')
    fns = glob(noise_eval_dir + '/*.csv')
    print(f'Found {len(fns)} examples to process...')

    uid_filter = None
    if args.only_eval:
        # Only run evaluation code for the 1,000 set aside for evaluation (just faster)
        uid_filter = set(pd.read_csv(
            os.path.join(args.input_dir, args.target, 'high_quality', 'eval_examples.csv'))['uid'])

    if args.debug:
        args.cpu_frac = -1

    if args.cpu_frac == -1:
        outputs = list(itertools.chain(*list(map(lambda fn: process(
            fn, noise_nsp_dir, noise_ent_eval_dir, uid_filter), fns))))
    else:
        outputs = list(itertools.chain(*list(
            p_uimap(lambda fn: process(fn, noise_nsp_dir, noise_ent_eval_dir, uid_filter), fns, num_cpus=args.cpu_frac))))
    outputs = pd.DataFrame(outputs)
    mini_str = '_mini' if args.debug else ''
    out_dir = os.path.join(data_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_fn = os.path.join(out_dir, f'results{mini_str}.csv')
    print(f'Saved results to {out_fn}')
    outputs.to_csv(out_fn, index=False)

    stats = ['fake_frac', 'nsp', 'bert_bs_recall', 'bert_bs_precision', 'electra_bs_recall', 'electra_bs_precision']
    stats += list(PERTURB_ENT_COLS)
    perturbs = ['shuffle_orderliness', 'span_remove', 'ent_add', 'ent_remove']

    original = outputs[outputs['version'] == 'original']
    perturbed = outputs[outputs['version'] == 'perturbed']

    for stat in stats:
        try:
            original_vals = original[stat].dropna()
            perturbed_vals = perturbed[stat].dropna()
            original_mean = 'n/a' if len(original_vals) == 0 else original_vals.mean()
            perturbed_mean = 'n/a' if len(perturbed_vals) == 0 else perturbed_vals.mean()
            print(stat, original_mean, perturbed_mean)
        except KeyError:
            print(f'Column={stat} not in dataframe.')

    correl_cols = ['fake_frac', 'bert_bs_precision', 'bert_bs_recall', 'electra_bs_recall', 'electra_bs_precision',
                   'nsp', 'local_halluc_frac', 'local_ent_f1', 'shuffle_orderliness', 'span_remove', 'ent_add',
                   'ent_remove']
    correl_cols = list(set(correl_cols).intersection(set(list(outputs.columns))))
    num_df = outputs[correl_cols]
    corr_table = num_df.corr(method='pearson')

    fig, ax = plt.subplots()
    sns.heatmap(num_df.corr(method='pearson'), annot=True, fmt='.1f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    plt.savefig(os.path.join(out_dir, 'correlation.png'), bbox_inches='tight', pad_inches=0)
