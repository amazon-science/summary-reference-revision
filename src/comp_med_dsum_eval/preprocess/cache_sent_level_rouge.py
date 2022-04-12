# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import itertools

from scipy.stats import rankdata
import pandas as pd
from p_tqdm import p_uimap


from comp_med_dsum_eval.eval.rouge import calc_rouge, preprocess_sentence
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent, sents_from_html, get_sent_idxs, get_attr


def compute_sent_level_rouge(record):
    source = record['source']
    target = record['target']

    target_str = preprocess_sentence(remove_tags_from_sent(target))
    source_sents = [preprocess_sentence(remove_tags_from_sent(x)) for x in sents_from_html(source)]
    sent_ids = get_sent_idxs(source)
    assert len(sent_ids) == len(source_sents)
    outputs = []
    rouge_scores = [calc_rouge(source_sent, target_str) for source_sent in source_sents]
    for sent_id, rouge_result in zip(sent_ids, rouge_scores):
        row = {'example_id': record['example_id'], 'sent_id': sent_id}
        for rouge_type, score in rouge_result.items():
            row[f'{rouge_type}_precision'] = score.precision
            row[f'{rouge_type}_recall'] = score.precision
            row[f'{rouge_type}_f1'] = score.fmeasure
        outputs.append(row)
    return outputs


def annotate(record, rouge_scores):
    rouge_scores.sort_values(by='sent_id', inplace=True)
    rouge_scores['agg_scores'] = rouge_scores[['rouge1_f1', 'rouge2_f1', 'rougeL_f1']].mean(axis=1)
    rouge_scores['rank'] = rankdata(-rouge_scores['agg_scores'], method='ordinal')
    source = record['source']

    sent_score_map = dict(zip(rouge_scores['sent_id'], rouge_scores['agg_scores']))
    sent_rank_map = dict(zip(rouge_scores['sent_id'], rouge_scores['rank']))

    tagged_tps = []
    tps = source.split('<SEP>')
    for tp_idx, tp in enumerate(tps):
        if tp_idx > 0 and tp.startswith('<s'):
            sent_idx = int(get_attr(tp, 'idx'))
            score = round(sent_score_map[sent_idx] * 1000)
            rank = sent_rank_map[sent_idx]
            tp = f'<s idx={str(sent_idx)} score={score} rank={rank}>'
        tagged_tps.append(tp)
    tagged_source = '<SEP>'.join(tagged_tps)
    record['source'] = tagged_source
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Truncate Dataset')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--version', default='original', choices=[
        'original',
        'revised_balanced',
        'revised_max_coverage',
        'revised_extractive'
    ])
    parser.add_argument('--reviser_experiment', default='yay', choices=[
        'yay', 'no_mask', 'no_neg', 'no_redress', 'no_same_sum',
    ])

    args = parser.parse_args()

    data_suffix = '' if args.version == 'original' else f'_{args.reviser_experiment}_{args.version}'
    data_dir = os.path.join(args.input_dir, args.target)
    mini_str = '_mini' if args.debug else ''
    if args.debug:
        args.cpu_frac = -1
    in_fn = os.path.join(data_dir, f'summary_dataset{data_suffix}{mini_str}.csv')
    print(f'Loading data from {in_fn}')
    df = pd.read_csv(in_fn)
    df = df.sort_values(by='source', key=lambda x: x.str.len())
    records = df.to_dict('records')

    rouge_out_fn = os.path.join(data_dir, f'source_sent_rouge_scores{data_suffix}{mini_str}.csv')
    annotated_out_fn = os.path.join(data_dir, f'summary_dataset_rouge_annotated{data_suffix}{mini_str}.csv')
    if args.cpu_frac == -1:
        outputs = list(itertools.chain(*list(map(compute_sent_level_rouge, records))))
    else:
        outputs = list(itertools.chain(*list(p_uimap(compute_sent_level_rouge, records, num_cpus=args.cpu_frac))))
    print('Done processing! Now collecting records into a dataframe.')
    rouge_scores = pd.DataFrame(outputs)
    print(f'Saving ROUGE scores for {len(rouge_scores)} source sentences to {rouge_out_fn}')
    rouge_scores.to_csv(rouge_out_fn, index=False)
    ex_rouge = dict(tuple(rouge_scores.groupby(by='example_id')))
    if args.cpu_frac == -1:
        outputs = list(map(lambda record: annotate(record, ex_rouge[record['example_id']]), records))
    else:
        outputs = list(p_uimap(lambda record: annotate(
            record, ex_rouge[record['example_id']]), records, num_cpus=args.cpu_frac))

    print('Done processing! Now collecting records into a dataframe.')
    output_df = pd.DataFrame(outputs)
    print(f'Saving {len(output_df)} ROUGE score annotated examples to {annotated_out_fn}')
    output_df.to_csv(annotated_out_fn, index=False)
