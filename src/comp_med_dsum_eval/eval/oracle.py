# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import argparse
import pandas as pd
from p_tqdm import p_uimap
from nltk import word_tokenize

from comp_med_dsum_eval.eval.rouge import calc_rouge
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent, sents_from_html, get_attr


def score_sents(record):
    outputs = []
    source_tps = record['source'].split('<SEP>')
    reference = ' '.join(word_tokenize(remove_tags_from_sent(record['target'])))
    sent_idx = -1
    for idx, tp in enumerate(source_tps):
        if tp.startswith('<s'):
            sent_idx += 1
            sent_id = int(get_attr(tp, 'idx'))
            assert sent_id == sent_idx
            sent = ' '.join(word_tokenize(remove_tags_from_sent(source_tps[idx + 1])))
            rouge = calc_rouge(sent, reference)
            outputs.append({
                'example_id': record['example_id'],
                'sent_id': sent_id,
                'rouge': rouge
            })
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute ROUGE targets against ground-truth reference...')
    parser.add_argument('--input_dir', default='/efs/griadams/hpi/')
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    # parser.add_argument('--granularity', default='sentence', choices=['sentence', 'section', 'note'])

    args = parser.parse_args()

    in_fn = os.path.join(args.input_dir, 'summary_dataset_ent.csv')
    print(f'Loading in data from {in_fn}')
    data_df = pd.read_csv(in_fn)
    if args.cpu_frac == -1:
        outputs = list(itertools.chain(*list(map(score_sents, data_df.to_dict('records')))))
    else:
        outputs = list(itertools.chain(*list(p_uimap(score_sents, data_df.to_dict('records'), num_cpus=args.cpu_frac))))
    outputs = pd.DataFrame(outputs)
    out_fn = os.path.join(args.input_dir, 'rouge_stats.csv')
    outputs.to_csv(out_fn, index=False)
