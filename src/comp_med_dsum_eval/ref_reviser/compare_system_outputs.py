# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from comp_med_dsum_eval.ref_reviser.dataset import remove_tags_from_sent


EXPERIMENTS = [
    'yay_repeat',
    'no_neg',
    'no_mask',
    'no_redress',
    'no_same_sum',
    'raw'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compare System Revision Outputs')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc')

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)
    annotated_dir = os.path.join(data_dir, 'revise', 'annotated')
    eval_dir = os.path.join(data_dir, 'revise', 'eval')
    output_dir = os.path.join(data_dir, 'revise', 'system_comparisons')
    os.makedirs(output_dir, exist_ok=True)

    output_str = []
    experiment_dirs = [os.path.join(annotated_dir, exp) for exp in EXPERIMENTS]
    experiment_ids = [x.split('/')[-1].replace('.csv', '') for x in glob(experiment_dirs[0] + '/*.csv')]
    for ex_id in tqdm(experiment_ids, total=len(experiment_ids)):
        output_str = [f'Example ID -> {ex_id}']
        exp_dfs = [dict(tuple(pd.read_csv(os.path.join(experiment_dir, f'{ex_id}.csv')).groupby('target_sent_idx')))
                   for experiment_dir in experiment_dirs]
        target_sent_idxs = list(exp_dfs[0].keys())
        for target_sent_idx in target_sent_idxs:
            output_str.append(f'Target Sent Idx: {target_sent_idx}')
            output_str.append('Target Sent Annotated: ' + exp_dfs[0][target_sent_idx]['text_original_annotated'].tolist()[0])
            target = exp_dfs[0][target_sent_idx]['text_original_annotated'].tolist()[0]
            target_clean = remove_tags_from_sent(target)
            output_str.append('Target Sent: ' + target_clean)
            output_str.append('Target Sent Annotated: ' + target)
            output_str.append('Source to Target Coverage: ' + str(exp_dfs[0][target_sent_idx]['source_to_target_coverage']))
            output_str.append('Hallucinations: ' + str(exp_dfs[0][target_sent_idx]['num_hallucinations']) +
                              '/' + str(exp_dfs[0][target_sent_idx]['target_ent_num']))
            for exp_name, exp_df in zip(EXPERIMENTS, exp_dfs):
                exp_out = exp_df.get(target_sent_idx, None)
                if exp_out is not None:
                    for record in exp_out.to_dict('records'):
                        if record['source_extract_code'] >= 5 and record['source_extract_code'] <= 10:
                            output_str.append(f'Experiment: {exp_name}')
                            output_str.append('Source Extract Code: ' + str(record['source_extract_code']))
                            output_str.append('Input Extract Code: ' + str(record['input_extract_code']))
                            output_str.append('Gen Input F1: ' + str(round(record['gen_input_sim'], 2)))
                            output_str.append('Gen Context F1: ' + str(round(record['gen_context_f1'], 2)))
                            output_str.append('Gen Context Precision: ' + str(round(record['gen_context_cov'], 2)))
                            output_str.append('Gen Context Recall: ' + str(round(record['gen_context_prec'], 2)))
                            output_str.append('Fake Score Fraction: ' + str(round(record['fake_frac'])))
                            output_str.append('Number of entities: ' + str(record['num_ents']))
                            output_str.append('Hallucination Rate: ' + str(round(record['global_halluc_frac'], 2)))
                            output_str.append('Faithful Adjusted Coverage: ' + str(record['num_ents']))
                            output_str.append('Prediction: ' + str(record['text_annotated']))
                            output_str.append('')
                output_str.append('')
            output_str.append('-' * 50)
        out_fn = os.path.join(output_dir, f'{ex_id}.txt')
        with open(out_fn, 'w') as fd:
            fd.write('\n'.join(output_str))
