# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os
import regex as re

import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

from comp_med_dsum_eval.eval.entailment import load_entail_scorer
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments
from comp_med_dsum_eval.preprocess.constants import HTML_REGEX_NO_SPACE
from comp_med_dsum_eval.eval.rouge import preprocess_sentence


def compute_entailments(df, device, entail_scorer, max_batch_size=128, hypothesis_cols=None):
    entail_inputs = []
    cols = []
    feats = []
    example_id = df['example_id'].tolist()[0]
    seen_predicted_sents = set()
    for record in df.to_dict('records'):
        context = ' '.join([x for x in re.split(HTML_REGEX_NO_SPACE, record['context'])
                            if len(x) > 0 and not x.startswith('<s') and not x.startswith('</s')])
        context_toks = [x.lower() for x in preprocess_sentence(context).split(' ')]
        for col in hypothesis_cols:
            sent2 = record[col]
            if col == 'target_sent' and sent2 in seen_predicted_sents:
                continue
            if col == 'target_sent':
                seen_predicted_sents.add(sent2)
            hypo_toks = [x.lower() for x in preprocess_sentence(sent2).split(' ')]
            perturb_frags = parse_extractive_fragments(context_toks, hypo_toks, remove_stop=True)
            perturb_frags['source_extract_code'] = None if col == 'target_sent' else record['source_extract_code']
            perturb_frags['input_extract_code'] = None if col == 'target_sent' else record['input_extract_code']
            perturb_frags['predict_sent_idx'] = record['predict_sent_idx']
            perturb_frags['context'] = record['context'].replace(' idx=-1', '')
            entail_inputs.append(f'mednli: sentence1: {context} sentence2: {sent2}')
            cols.append(col)
            feats.append(perturb_frags)

    n = len(entail_inputs)
    text_batches = [list(x) for x in np.array_split(np.arange(n), round(n // max_batch_size) + 1)]
    text_batches = [x for x in text_batches if len(x) > 0]
    encoding = entail_scorer['tokenizer'](
        entail_inputs, padding='max_length', truncation=True, max_length=256, return_tensors='pt'
    )
    input_ids, attention_masks = encoding['input_ids'], encoding['attention_mask']
    entail_labels = []
    with torch.no_grad():
        for batch_idxs in text_batches:
            batch_input_ids = input_ids[batch_idxs].to(device)
            batch_attention_mask = attention_masks[batch_idxs].to(device)
            with torch.no_grad():
                outputs = entail_scorer['model'].generate(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask,
                    max_length=8,
                    early_stopping=True
                )
                entail_labels += list(map(
                    lambda x: entail_scorer['tokenizer'].decode(
                        x, skip_special_tokens=True, clean_up_tokenization_spaces=True),
                    outputs)
                )
    outputs = []
    for col, feat, label in zip(cols, feats, entail_labels):
        outputs.append({
            'example_id': example_id,
            'predict_sent_idx': feat['predict_sent_idx'],
            'version': 'original' if col == 'target_sent' else 'corrected',
            'entail': label,
            'coverage': feat['coverage'],
            'density': feat['density'],
            'input_extract_code': feat['input_extract_code'],
            'source_extract_code': feat['source_extract_code'],
            'context': feat['context'],
        })
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Measure Post-Hoc Corrected vs Predicted Effective Faithfulness')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--summary_experiment', default='longformer_16384_full')  # Get results from revise outputs
    parser.add_argument('--gpu_device', default=None, type=int)

    args = parser.parse_args()

    # free_gpus = get_free_gpus()
    # args.gpu_device = free_gpus[0] if args.gpu_device is None else args.gpu_device
    #
    # # revise / weights
    # data_dir = os.path.join(args.input_dir, args.target)
    #
    # edit_dir = os.path.join(data_dir, 'revise', 'output', args.summary_experiment)
    # edit_fns = glob(os.path.join(edit_dir, '*.csv'))
    #
    # hypothesis_cols = ['prediction', 'target_sent']
    # entail_scorer = load_entail_scorer(device=args.gpu_device)
    # outputs = []
    # for edit_fn in tqdm(edit_fns, total=len(edit_fns)):
    #     df = pd.read_csv(edit_fn)
    #     outputs += compute_entailments(
    #         df, device=args.gpu_device, entail_scorer=entail_scorer, hypothesis_cols=hypothesis_cols
    #     )
    #
    # outputs = pd.DataFrame(outputs)
    # outputs.to_csv('~/extractive.csv', index=False)

    outputs = pd.read_csv('~/extractive.csv')
    outputs = outputs.assign(is_entail=outputs['entail'].apply(lambda x: 1 if x == 'entailment' else 0))
    correct = outputs[outputs['version'] == 'corrected']

    def compute_priority(row):
        priority = -row['coverage']  # Prefer abstractive
        if row['entail'] == 'entailment':
            priority += 1  # Prefer entailed
        return priority


    # Select reasonable range for target extractiveness: 50-90%
    outputs = outputs[(outputs['source_extract_code'].isnull() | outputs['source_extract_code'].between(5, 9))]
    correct = correct.assign(priority=outputs.apply(compute_priority, axis=1)).sort_values(by='priority', ascending=False)
    best_correct = correct.drop_duplicates(subset=['example_id', 'predict_sent_idx'])
    predicted = outputs[outputs['version'] == 'original']
    print('Bucket,Predicted Frac Entailed, Corrected Fraction Entailed')
    versions = ['original', 'corrected']
    gran = 0.1
    for bucket in np.arange(0, 1, gran):
        fracs = []
        supports = []
        for version in versions:
            sub_df = outputs[outputs['version'] == version]
            # Ask output to be atleast as extractive
            # if version == 'corrected':
            #     sub_df = sub_df[sub_df['source_extract_code'] >= sub_df['input_extract_code']]
            rel_df = sub_df[sub_df['coverage'].between(bucket, bucket + gran)]
            supports.append(len(rel_df))
            num_entail = len(rel_df[rel_df['entail'] == 'entailment'])
            overall = len(rel_df)
            frac_entail = 0 if overall == 0 else num_entail / overall
            fracs.append(frac_entail)

        best_sub_df = best_correct[best_correct['coverage'].between(bucket, bucket + gran)]
        num_entail = len(best_sub_df[best_sub_df['entail'] == 'entailment'])
        overall = len(best_sub_df)
        best_frac_entail = 0 if overall == 0 else num_entail / overall
        print(f'{round(bucket, 2)}-{round(bucket + gran, 2)},{round(fracs[0], 3)},'
              f'{round(fracs[1], 3)},{round(best_frac_entail, 3)}')  # ,{supports[0]},{supports[1]}')
