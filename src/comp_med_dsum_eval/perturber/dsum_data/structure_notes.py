# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import pandas as pd
from p_tqdm import p_uimap
import spacy

from comp_med_dsum_eval.preprocess.sec_tag.main import section_tagger_init
from comp_med_dsum_eval.preprocess.mimic_utils import add_note_id
from comp_med_dsum_eval.preprocess.structure_notes import decorate
from comp_med_dsum_eval.preprocess.collect_dataset import add_ids


def decorate_add_idxs(record, sentencizer):
    record['TEXT'] = add_ids(decorate(record, sentencizer, None)['TEXT'])
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to structure discharge summary notes.')
    parser.add_argument('--output_dir', default='/efs/griadams')
    parser.add_argument('--input_dir', default='/efs/hanchins/internship')
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.cpu_frac = -1

    mini_str = '_mini' if args.debug else ''
    output_dir = os.path.join(os.path.join(args.output_dir, 'dsum'))
    os.makedirs(output_dir, exist_ok=True)
    in_fn = os.path.join(args.input_dir, f'NOTEEVENTS_DischargeSummary{mini_str}.csv')
    discharge_summary_df = pd.read_csv(in_fn)

    section_tagger_init()
    print('Loading Spacy...')
    sentencizer = spacy.load('en_core_sci_sm', disable=['ner', 'parser', 'lemmatizer'])
    sentencizer.add_pipe('sentencizer')
    add_note_id(discharge_summary_df)
    records = discharge_summary_df.to_dict('records')
    if args.cpu_frac == -1:
        augmented_records = pd.DataFrame(list(map(
            lambda record: decorate_add_idxs(record, sentencizer), records)))
    else:
        augmented_records = pd.DataFrame(list(p_uimap(
            lambda record: decorate_add_idxs(record, sentencizer), records, num_cpus=args.cpu_frac)))
    out_fn = os.path.join(output_dir, 'dsum_tagged.csv')
    print(f'Saving {len(augmented_records)} augmented notes to {out_fn}')
    augmented_records.to_csv(out_fn, index=False)

    mini_df = augmented_records.sample(n=128, random_state=1992, replace=False)
    out_fn = os.path.join(output_dir, 'dsum_tagged_mini.csv')
    print(f'Saving {len(mini_df)} augmented notes to {out_fn}')
    mini_df.to_csv(out_fn, index=False)
