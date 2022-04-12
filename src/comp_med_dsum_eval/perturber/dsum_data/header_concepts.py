# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import os
import regex as re

import argparse
import pandas as pd
from tqdm import tqdm

from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_sections, remove_empty

REMOVE_CONCEPTS = {
    # Admin / billing codes
    'service', 'code_status', 'cpt_code'
    # PHI
    'age', 'race', 'language', 'gender', 'weight', 'address',
    'phone_number', 'work_phone_number', 'cell_phone_number', 'home_phone_number',
    # Physician information
    'dictating_physician', 'primary_physician', 'physician', 'attending_physician', 'referring_physician',
    'requesting_physician', 'providers'
    # Data
    'laboratory_and_radiology_data', 'laboratory_data', 'valvular_data', 'pathologic_data', 'objective_data',
    'psychological_data',
    # Date / Time
    'date_of_discharge', 'date', 'date_transcribed', 'date_dictated', 'admission_date', 'date_of_procedure',
    'date_of_birth', 'date_of_admission', 'report_date', 'date_of_death', 'estimated_due_date',
    'date_time', 'time', 'total_time', 'time_of_birth',
}


def extract_section_concepts(text):
    return re.findall(r'con=([^\s]+)', text)


def remove_wanted_sections(text):
    tps = remove_sections(text, cons_to_remove=REMOVE_CONCEPTS)
    if tps is None:
        return None
    return '<SEP>'.join(remove_empty(tps.split('<SEP>'), 'h'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to structure discharge summary notes.')
    parser.add_argument('--input_dir', default='/efs/griadams/dsum')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--mode', default='gen', choices=['gen', 'remove_sections'])

    args = parser.parse_args()

    mini_str = '_mini' if args.debug else ''
    in_fn = os.path.join(args.input_dir, f'dsum_tagged{mini_str}.csv')
    df = pd.read_csv(in_fn)

    texts = df['TEXT'].tolist()
    if args.mode == 'gen':
        counts = defaultdict(int)

        for text in tqdm(texts, total=len(texts)):
            concepts = extract_section_concepts(text)
            for concept in concepts:
                counts[concept] += 1

        out_df = pd.DataFrame(counts.items(), columns=['concept', 'count'])
        out_df.sort_values(by='count', ascending=False, inplace=True)
        out_fn = os.path.join(args.input_dir, 'section_counts.csv')
        print(f'Saving {len(out_df)} section header counts to {out_fn}')
        out_df.to_csv(out_fn, index=False)
    else:
        df['TEXT'] = df['TEXT'].apply(remove_wanted_sections)
        df.dropna(subset=['TEXT'], inplace=True)
        out_fn = os.path.join(args.input_dir, f'dsum_tagged_filt{mini_str}.csv')
        print(f'Saving {len(df)} examples to {out_fn}')
        df.to_csv(out_fn, index=False)
