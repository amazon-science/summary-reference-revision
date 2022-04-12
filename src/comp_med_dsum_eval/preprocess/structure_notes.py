# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from collections import defaultdict
from tqdm import tqdm

import argparse
import pandas as pd
from p_tqdm import p_uimap
import spacy

from comp_med_dsum_eval.preprocess.mimic_utils import clean_mimic, create_note_tok, remove_identifiers
from comp_med_dsum_eval.preprocess.sec_tag.main import process_report, section_tagger_init
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_empty_sections, safe_html, section_split
from comp_med_dsum_eval.preprocess.sent_segment.main import parse_sentences_spacy
from comp_med_dsum_eval.preprocess.variations import mapping_variations


def tag_target(text, sentencizer, target_headers):
    text_pieces = [x for x in re.split(r'(' + '|'.join(target_headers) + ')', text) if len(x.strip()) > 0]
    if len(text_pieces) == 1:
        return None

    headers = []
    seen_header = False
    output_pieces = []
    for tp in text_pieces:
        if tp in target_headers:
            headers.append(safe_html(tp))
            seen_header = True
        else:
            if not seen_header:
                print('Error in processing headers. Skipping.')
                return None
            header_str = ';'.join(headers)
            target_header_html = f'<h con=target raw={header_str}>'
            output_pieces.append(target_header_html)
            sents = parse_sentences_spacy(tp, sentencizer)
            for sent_idx, sent in enumerate(sents):
                output_pieces.append('<s>')
                output_pieces.append(sent)
                output_pieces.append('</s>')
    output_pieces.append('</h>')
    return output_pieces


def tag_text(text, sentencizer, remove_empty_sub_sections=False, prepend_sub_sections=False, verbose=False):
    # BRIEF HOSPITAL COURSE     : patient ... -> BRIEF HOSPITAL COURSE: patient ...
    text = re.sub(r'\s+:', ':', text)
    # remove any HTML tags
    text = re.sub(r' *<\/?[^>]*> *', ' ', text)
    # also remove '< >' because this messes with our crude HTML parser
    text = re.sub(r' ?[<>]+ ?', ' ', text)
    text = remove_identifiers(text)
    text = text.strip()
    if len(text) == 0:
        return None
    sections, section_pieces = process_report(text, sentencizer, verbose=verbose)
    if sections is None:
        assert section_pieces is None
        return None
    output_pieces = []
    for section, section_piece in zip(sections, section_pieces):
        if len(section_piece) == 0:
            continue
        treecode_str = '.'.join([str(i) for i in section.treecode_list])
        concept = safe_html(section.concept.lower())
        synonym = safe_html(section.synonym)
        raw = safe_html(section.raw)
        assert len(raw) > 0
        section_str = f'<h con={concept} syn={synonym} tree={treecode_str} raw={raw}>'
        assert '<' not in section_str[1:-1] and '>' not in section_str[1:-1]
        output_pieces.append(section_str)
        subsection_pieces, is_header_arr = section_split(section_piece)
        open_p = False
        for sub_idx, (sub_piece, is_header) in enumerate(zip(subsection_pieces, is_header_arr)):
            sub_piece = re.sub(r'\s+', ' ', sub_piece).strip()
            if len(sub_piece) == 0:
                continue
            if is_header:
                is_next_header = sub_idx < len(is_header_arr) - 1 and is_header_arr[sub_idx + 1]
                if remove_empty_sub_sections and is_next_header:
                    continue
                if open_p:
                    output_pieces.append('</p>')
                p_name = safe_html(sub_piece)
                assert '<' not in p_name and '>' not in p_name
                output_pieces.append(f'<p name={p_name}>')
                open_p = True
            else:
                sents = parse_sentences_spacy(sub_piece, sentencizer)
                for sent_idx, sent in enumerate(sents):
                    if prepend_sub_sections and sent_idx == 0 and sub_idx > 0 and is_header_arr[sub_idx - 1]:
                        sent = subsection_pieces[sub_idx - 1].strip() + ' ' + sent
                    output_pieces.append('<s>')
                    output_pieces.append(sent)
                    output_pieces.append('</s>')
        if open_p:
            output_pieces.append('</p>')
        output_pieces.append('</h>')
    return output_pieces


def decorate(record, sentencizer, target_headers, text_col='TEXT'):
    text = clean_mimic(record[text_col])
    if text_col == 'TEXT':
        output_pieces = tag_text(text, sentencizer, remove_empty_sub_sections=True, prepend_sub_sections=True)
    else:
        output_pieces = tag_target(text, sentencizer, target_headers)
    if output_pieces is not None:
        if 'note_id' in record:
            note_type_str = create_note_tok(record['CATEGORY'], prefix=False)
            opening_tag = f"<d note_id={record['note_id']} note_type={note_type_str} " \
                          f"patient_id={record['SUBJECT_ID']} visit_id={record['HADM_ID']}>"
        else:
            opening_tag = f"<d patient_id={record['SUBJECT_ID']} visit_id={record['HADM_ID']}>"
        output_pieces.insert(0, opening_tag)
        output_pieces.append('</d>')
        # remove empty sections
        output_pieces = remove_empty_sections(output_pieces)
        output_str = '<SEP>'.join(output_pieces)
        record[text_col] = output_str
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Structuring notes by section & sentence.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--dtypes', default='source,target', choices=['source', 'target', 'source,target'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, args.target)
    os.makedirs(input_dir, exist_ok=True)

    section_tagger_init()
    print('Loading Spacy...')
    sentencizer = spacy.load('en_core_sci_sm', disable=['ner', 'parser', 'lemmatizer'])
    sentencizer.add_pipe('sentencizer')

    variation = 'history_of_present_illness' if args.target == 'hpi' else 'brief_hospital_course'
    target_headers = [x[0] for x in mapping_variations[variation]]

    for dtype in args.dtypes.split(','):
        in_fn = os.path.join(input_dir, '{}_notes.csv'.format(dtype))
        out_fn = os.path.join(input_dir, '{}_notes_tagged.csv'.format(dtype))
        print('Reading in {}'.format(in_fn))
        note_df = pd.read_csv(in_fn)
        records = note_df.to_dict('records')
        print('Processing {} notes'.format(len(note_df)))
        text_col = 'target' if 'target' in records[0] else 'TEXT'
        if args.cpu_frac == -1:
            augmented_records = pd.DataFrame(list(map(
                lambda record: decorate(record, sentencizer, target_headers, text_col=text_col), records)))
        else:
            augmented_records = pd.DataFrame(list(p_uimap(
                lambda record: decorate(
                    record, sentencizer, target_headers, text_col=text_col), records, num_cpus=args.cpu_frac)))
        print(f'Saving {len(augmented_records)} augmented notes to {out_fn}')
        augmented_records.to_csv(out_fn, index=False)

        if dtype == 'source':
            print('Storing source note type counts for corpus.')
            source_notes = augmented_records['TEXT'].dropna().tolist()
            counter = defaultdict(int)
            for note in tqdm(source_notes):
                note_types = re.findall(r'note_type=([^ ]+)', note)
                for nt in note_types:
                    counter[nt] += 1

            output_df = []
            for k, ct in counter.items():
                output_df.append({
                    'note_type': k,
                    'count': ct
                })

            note_types_df = pd.DataFrame(note_types)
            note_type_fn = os.path.join(args.input_dir, 'note_types.csv')
            output_df.to_csv(note_type_fn, index=False)
