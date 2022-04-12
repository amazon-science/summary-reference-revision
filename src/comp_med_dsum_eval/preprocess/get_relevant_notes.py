# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Dataset from MIMIC III.

Usage:
    generate_dataset_for_target.py [options]
    generate_dataset_for_target.py --help

Options:
    -h --help                    show this help message and exit
    -o --output=<dir>            output dir [default: /efs/hanchins/past_medical_history_dataset]
    -i --input=<dir>             input dir [default: /efs/hanchins/]
    -overwrite                   Whether or not to over-write cached output files
"""
import argparse
import pandas as pd
import json
import os
from p_tqdm import p_uimap
import itertools
from tqdm import tqdm
from comp_med_dsum_eval.preprocess.mimic_utils import add_note_id
from comp_med_dsum_eval.preprocess.variations import mapping_variations
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../generate_dataset_for_target.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# progress bar for pandas
tqdm.pandas()


def is_in(s, variations):
    """
    check if string is in variation
    :param s: str
    :param variations: List[Tuple[str, int]], candidate variation
    :return: bool, whether s is in variation
    """
    return any([v in s for v, count in variations])


def identify_target_section_from_text(list_of_text, variations, other_variations):
    """
    Extract target span.
    For each line in text, check if it contains the target section header
    Extract everything starting from the target section header to the next non-target section header (not included)
    :param list_of_text: List[str], list of raw text
    :param variations: Set[str], set of target variations
    :param other_variations: Set[str], set of non-target variations
    :return: List[str], list of text to be process
    """
    def section_header_in_line(line, section_header_variations):
        return line.strip().lower() in section_header_variations or \
               (':' in line and line.strip().split(':')[0].lower() + ':' in section_header_variations)

    text_to_be_process = []

    # identify target section from text
    for text in tqdm(list_of_text):
        span = []
        start = False
        for line in text.split('\n'):
            if section_header_in_line(line, variations):
                span.append(line)
                start = True
            elif start and section_header_in_line(line, other_variations):
                break
            elif start:
                span.append(line)
        text_to_be_process.append(' \n '.join(span))
    return text_to_be_process


def get_target_span(discharge_summary_df, variations, all_variations):
    """
    Extract target span (target medical section) from discharge summary
    :param discharge_summary_df: pd.DataFrame, each row is a discharge summary.
    :param variations: List[str, int], variations of the target medical section.
    :param all_variations: Set[str], all variations that resembles possible section headers.
    :return: pd.DataFrame, each row is a represented discharge summary,
             and row.target is the text of the target medical section
    """
    # an encounter can have multiple discharge summary, select the earliest one that is a Report type
    discharge_summary_df = discharge_summary_df.sort_values(['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])
    representative_discharge_summary_df = discharge_summary_df.loc[discharge_summary_df.DESCRIPTION == 'Report',
                                                                   ['SUBJECT_ID', 'HADM_ID', 'DESCRIPTION', 'CHARTDATE',
                                                                    'TEXT']].groupby('HADM_ID').agg('first')

    # check if the discharge summary text contains the target medical section, subsample accordingly
    tf = representative_discharge_summary_df.TEXT.apply(lambda x: is_in(x, variations))
    valid_representative_discharge_summary_df = representative_discharge_summary_df.loc[tf]

    # lowercased target sections in set
    variations_set = {v.strip().lower() for v, count in variations}

    # extract target medical section span
    valid_representative_discharge_summary_df['target'] = identify_target_section_from_text(
        valid_representative_discharge_summary_df.TEXT,
        variations_set,
        all_variations - variations_set)

    # filter out discharge summary with empty target section span
    valid_representative_discharge_summary_df = valid_representative_discharge_summary_df.loc[
                                                ~(valid_representative_discharge_summary_df.target.apply(len) == 0)]
    return valid_representative_discharge_summary_df


def load_data(input_dir):
    """
    load data from input dir and convert datetime field to correct format
    :param input_dir: str, input directory
    :return: dataframes and section header variations
    """
    all_notes = pd.read_csv(os.path.join(input_dir, 'NOTEEVENTS.csv'))
    discharge_summary_df = all_notes[(all_notes['CATEGORY'] == 'Discharge summary') & (all_notes['DESCRIPTION'] == 'Report')]

    def is_admission_note(note_description):
        return 'admi' in note_description.lower()
    all_visits = len(all_notes['HADM_ID'].unique())
    viable_hadm_ids = all_notes[all_notes['DESCRIPTION'].apply(is_admission_note)]['HADM_ID'].unique().tolist()
    viable_df = pd.DataFrame({'HADM_ID': viable_hadm_ids})
    print(f'{len(viable_df)}/{all_visits} visits with an admission note')
    discharge_summary_df['CHARTDATE'] = pd.to_datetime(discharge_summary_df.CHARTDATE)
    # pre-computed candidate sections from regex
    with open(os.path.join(input_dir, 'all_candidate_sections_v2.json'), 'r') as fh:
        all_candidate_sections_sorted = json.load(fh)
    common_candidate_sections_sorted = [
        (section, count) for section, count in all_candidate_sections_sorted if count > 10
    ]
    all_variations_set = {v.strip().lower() for v, count in common_candidate_sections_sorted}
    return discharge_summary_df, all_notes, all_variations_set, viable_df


def filter_valid_notes(hadm_df, dsum_date):
    dsum_date = pd.to_datetime(dsum_date)
    return list(filter(
        lambda record: pd.to_datetime(record['CHARTDATE']) <= dsum_date and record['CATEGORY'] != 'Discharge summary',
        hadm_df.to_dict('records')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to extract relevant notes for HPI/BHC MIMIC dataset.')
    parser.add_argument('-overwrite', action='store_true', default=False,
                        help='Turn on to over-write cached intermediate files.  Otherwise, they will be used.')
    parser.add_argument('--output_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--input_dir', default='/efs/hanchins/internship')

    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, args.target)
    logger.info('Creating output dir {}...'.format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    logger.info('loading data...')
    discharge_summary_df, all_notes, all_variations_set, admission_df = load_data(args.input_dir)

    admit_fn = os.path.join(args.output_dir, 'viable_visits.csv')
    print(f'Saving {len(admission_df)} HADM_IDs (visit id) with an admission note to {admit_fn}')
    admission_df.to_csv(admit_fn, index=False)

    # get target span and prior notes
    logger.info('Get target span from discharge summary...')
    valid_rep_dsum_fn = os.path.join(output_dir, 'target_notes.csv')
    if os.path.exists(valid_rep_dsum_fn) and not args.overwrite:
        print(f'Loading {valid_rep_dsum_fn}...')
        valid_representative_discharge_summary_df = pd.read_csv(valid_rep_dsum_fn)
    else:
        target_key = 'history_of_present_illness' if args.target == 'hpi' else 'brief_hospital_course'
        print(f'Extracting {target_key} section from {len(discharge_summary_df)} discharge summaries')
        valid_representative_discharge_summary_df = get_target_span(
            discharge_summary_df, mapping_variations[target_key], all_variations_set)
        n = len(valid_representative_discharge_summary_df)
        print(f'Saving Valid Representative Discharge Summary DF ({n} rows) to {valid_rep_dsum_fn}')
        valid_representative_discharge_summary_df.to_csv(valid_rep_dsum_fn, index=True)

    hadm_ids = (valid_representative_discharge_summary_df['HADM_ID'].tolist()
                if 'HADM_ID' in list(valid_representative_discharge_summary_df.columns)
                else valid_representative_discharge_summary_df.index.tolist())
    hadm2date = dict(zip(hadm_ids, valid_representative_discharge_summary_df['CHARTDATE'].tolist()))

    logger.info('filter all notes down to prior notes of discharge summary...')
    prior_notes_fn = os.path.join(output_dir, 'source_notes.csv')
    if os.path.exists(prior_notes_fn) and not args.overwrite:
        print(f'Loading {prior_notes_fn}')
        valid_notes = pd.read_csv(prior_notes_fn)
    else:
        valid_hadm_ids = set(valid_representative_discharge_summary_df.index)
        print(f'Filtering notes for {len(valid_hadm_ids)} unique HADM_IDs')
        valid_notes = all_notes[all_notes['HADM_ID'].isin(valid_hadm_ids)]
        notes_by_hadm = list(tuple(valid_notes.groupby('HADM_ID')))
        prior_notes = pd.DataFrame(list(itertools.chain(*list(p_uimap(
            lambda x: filter_valid_notes(x[1], hadm2date[x[0]]), notes_by_hadm)))))
        assert len(prior_notes) > 0  # no mismatch in HADM_ID formatting
        add_note_id(prior_notes)
        print(f'Saving {len(prior_notes)} valid source notes prior to, or at, discharge to {prior_notes_fn}')
        prior_notes.to_csv(prior_notes_fn)

        notes_meta = prior_notes[['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'STORETIME', 'CATEGORY',
                                  'DESCRIPTION', 'CGID','ISERROR',]]
        meta_fn = os.path.join(output_dir, 'source_notes_meta.csv')
        notes_meta.to_csv(meta_fn, index=False)
