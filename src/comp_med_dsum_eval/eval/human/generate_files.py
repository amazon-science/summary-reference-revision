# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import numpy as np
import pandas as pd

from comp_med_dsum_eval.preprocess.collect_dataset import add_example_id
from comp_med_dsum_eval.eval.human.constants import NOTES_PATH


def sample_example_ids(df, size=50):
    start, end = round(len(df) * 0.1), round(len(df) * 0.9)
    df = df.assign(source_tok_ct=df['source'].apply(lambda x: len(x.split(' '))))
    valid_example_ids = df.sort_values(by='source_tok_ct')[start:end]['example_id'].tolist()
    return list(sorted(list(np.random.choice(valid_example_ids, size=size, replace=False))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating Human Evaluation Forms.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument(
        '--experiments',
        default='longformer_16384_full,long_high_only,long_revised_balanced,long_revised_extractive,long_revised_max_coverage'
    )
    parser.add_argument('--seed', default=1992, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    experiments = args.experiments.split(',')
    data_dir = os.path.join(args.input_dir, args.target)
    results_dir = os.path.join(data_dir, 'mimic_sum', 'results')

    meta_fn = os.path.join(data_dir, 'source_notes_meta.csv')
    print(f'Reading in source note meta information from {meta_fn}...')
    source_notes = pd.read_csv(meta_fn)
    add_example_id(source_notes)
    print('Grouping by Example ID...')
    exid2meta = dict(tuple(source_notes.groupby(by='example_id')))

    print(f'Reading in MIMIC-III notes from {NOTES_PATH}')
    notes = pd.read_csv(NOTES_PATH)
    row_id_to_text = dict(zip(notes['ROW_ID'], notes['TEXT']))

    experiment_fns = [
        os.path.join(results_dir, experiment, 'outputs.csv') for experiment in experiments
    ]
    outputs = list(map(pd.read_csv, experiment_fns))
    example_ids = sample_example_ids(outputs[0])
    out_dir = os.path.join(data_dir, 'human', 'annotations')
    os.makedirs(out_dir, exist_ok=True)

    oracle_summary_order = []
    for example_id in example_ids:
        out_fn = os.path.join(out_dir, f'{example_id}.txt')
        source_note_meta = exid2meta[example_id].sort_values(by='CHARTDATE')
        note_desc = source_note_meta['DESCRIPTION'].tolist()
        note_categories = source_note_meta['CATEGORY'].tolist()
        note_texts = [row_id_to_text[row_id] for row_id in source_note_meta['ROW_ID'].tolist()]
        output_str = ''
        for description, category, text in zip(note_desc, note_categories, note_texts):
            output_str += f'START OF NOTE TYPE: {category}\n'
            output_str += f'START OF NOTE DESCRIPTION: {description}\n\n'
            output_str += text.strip()
            output_str += f'\n\nEND OF NOTE TYPE: {category}\n'
            output_str += f'END OF NOTE TITLE: {description}\n\n'
        output_str += '-' * 100
        output_str += '\n\nSTART OF SUMMARIES\n\n' + '-' * 100 + '\n\n'
        # Shuffle order
        exp_order = np.arange(len(experiments))
        np.random.shuffle(exp_order)
        oracle_summary_order.append({
            'example_id': example_id,
            'oracle_summary': ','.join([experiments[i] for i in exp_order])
        })
        for ct_idx, exp_id in enumerate(exp_order):
            exp = outputs[exp_id]
            prediction = exp[exp['example_id'] == example_id]['prediction'].tolist()[0]
            output_str += prediction
            output_str += '\n\nRelevance:\nConsistency:\nFluency:\nCoherence:\n\n'
            output_str += '-' * 100 + '\n\n'
        output_str += 'END OF SUMMARIES\n'
        with open(out_fn, 'w') as fd:
            fd.write(output_str.strip())

    oracle_summary_order = pd.DataFrame(oracle_summary_order)
    oracle_fn = os.path.join(data_dir, 'human', 'summary_ordering.csv')
    print(f'Saving randomly shuffled experiment names for each Example ID to {oracle_fn}')
    oracle_summary_order.to_csv(oracle_fn, index=False)
