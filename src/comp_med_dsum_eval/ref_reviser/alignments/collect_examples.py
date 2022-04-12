# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from glob import glob
import ujson
import os

import argparse
import numpy as np
np.random.seed(1992)
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent
from comp_med_dsum_eval.ref_reviser.alignments.utils import keep_sent


def process(fn):
    with open(fn, 'r') as fd:
        examples = ujson.load(fd)
        rows = []
        for example in examples:
            row = {'example_id': example['example_id']}
            row['target'] = example['target']['sent']
            row['sent_idx'] = example['target']['alignment']['sent_idx']
            row.update(example['stats'])
            source = example['source']
            source_sents = source['sents']
            sent_idxs = source['sent_idxs']
            improvements = source['improvements']
            keep_sent_idxs = [sent_idx for sent_idx, imp in zip(sent_idxs, improvements) if keep_sent(imp)]
            source_sents_keep = [sent for sent_idx, sent in zip(sent_idxs, source_sents) if sent_idx in keep_sent_idxs]
            source_str = ' <s> '.join(source_sents_keep)
            row['source'] = source_str
            rows.append(row)
        return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregating individual examples for sentence-level analysis.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--max_n', default=None, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    if args.debug:
        args.cpu_frac = -1
        args.max_n = 128

    data_dir = os.path.join(args.input_dir, args.target)
    revise_dir = os.path.join(data_dir, 'revise', 'contexts')
    fns = glob(revise_dir + '/*.json')
    if args.max_n is not None and args.max_n < len(fns):
        fns = list(np.random.choice(fns, size=(args.max_n), replace=False))
        print(f'Randomly sampling {args.max_n} files to collect data for.')
    if args.cpu_frac == -1:
        outputs = list(itertools.chain(*list(tqdm(map(process, fns), total=len(fns)))))
    else:
        outputs = list(itertools.chain(*list(p_uimap(process, fns, num_cpus=args.cpu_frac))))

    df = pd.DataFrame(outputs)
    df = df.assign(
        missing_nums=df['target_nums'] - df['covered_nums'],
        ent_halluc_rate=1 - df['covered_ent_num'].div(df['target_ent_num']).replace(np.nan, 1),
        num_halluc_rate=1 - df['covered_nums'].div(df['target_nums']).replace(np.nan, 1)
    )
    df = df.assign(
        high_quality=(df['ent_halluc_rate'] == 0) & (df['source_to_target_coverage'] >= 0.75),
        high_quality_w_ent=(df['ent_halluc_rate'] == 0) & (df['source_to_target_coverage'] >= 0.75)
                           & (df['target_ent_num'] > 0),
    )

    high_quality_n = df['high_quality'].sum()
    high_quality_perc = round(high_quality_n / len(df) * 100)
    high_quality_n_w_ent = df['high_quality_w_ent'].sum()
    high_quality_perc_w_ent = round(high_quality_n_w_ent / len(df) * 100)
    print(f'{high_quality_perc}% of examples are high quality.')
    print(f'{high_quality_perc_w_ent}% of examples are high quality AND contain >=1 entity.')
    df.sort_values(by='source_to_target_coverage', ascending=False, inplace=True)

    num_df = df.select_dtypes(include='number')
    corr_table = num_df.corr(method='pearson')

    fig, ax = plt.subplots()
    sns.heatmap(num_df.corr(method='pearson'), annot=True, fmt='.1f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    image_fn = 'images/correlation.png'
    print(f'Saving image to {image_fn}')
    plt.savefig(image_fn, bbox_inches='tight', pad_inches=0)

    if not args.debug:
        out_fn = os.path.join(data_dir, 'revise', 'sents_w_context.csv')
        print(f'Saving {len(df)} examples to {out_fn}')
        df.to_csv(out_fn, index=False)
        out_fn = os.path.join(data_dir, 'revise', 'sent_quality.csv')
        df.loc[:, ~df.columns.isin(['source', 'target'])].to_csv(out_fn, index=False)

    df_mini = df.sample(n=min(128, len(df)), replace=False, random_state=1922)
    out_mini_fn = os.path.join(data_dir, 'revise', 'sents_w_context_mini.csv')
    print(f'Saving {len(df_mini)} examples to {out_mini_fn}')
    df_mini.to_csv(out_mini_fn, index=False)

    text_outputs = []
    for record in df.sample(n=min(1024, len(df)), replace=False).to_dict('records'):
        source = remove_tags_from_sent(record['source'])
        target = remove_tags_from_sent(record['target'])
        cov = record['source_to_target_coverage']
        ent_cov = str(record['covered_ent_num']) + '/' + str(record['target_ent_num'])
        num_cov = str(record['covered_nums']) + '/' + str(record['target_nums'])

        text_outputs.append(f'BHC Sentence: {target}')
        text_outputs.append(f'Relevant Context: {source}')
        text_outputs.append(f'BERT Coverage: {cov}')
        text_outputs.append(f'Entity Coverage: {ent_cov}')
        text_outputs.append(f'Number Coverage: {num_cov}')
        text_outputs.append('')
        text_outputs.append('-' * 50)
        text_outputs.append('')

    out_fn = os.path.join('sents_w_context.txt')
    with open(out_fn, 'w') as fd:
        fd.write('\n'.join(text_outputs))
