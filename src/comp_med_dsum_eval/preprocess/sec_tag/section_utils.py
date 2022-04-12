# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
import string

from comp_med_dsum_eval.preprocess.constants import *


def remove_empty(text_pieces, tag):
    remove_idxs = set()
    for i, elem in enumerate(text_pieces):
        if i > 0 and elem == '</{}>'.format(tag) and text_pieces[i - 1].startswith('<{}'.format(tag)):
            remove_idxs.add(i)
            remove_idxs.add(i - 1)
    output = []
    for i, piece in enumerate(text_pieces):
        if i in remove_idxs:
            continue
        output.append(piece)
    return output


def extract_entities_from_html(html_str):
    tps = html_str.split('<SEP>')
    entities = []
    note_id = ''
    section = ''
    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<h'):
            section = get_attr(tp, 'con')
        elif tp.startswith('<d') and 'note_id' in tp:
            note_id = get_attr(tp, 'note_id')
        if tp.startswith('<s'):
            sent_id = int(get_attr(tp, 'idx'))
            sub_tps = re.split(HTML_REGEX, tps[tp_idx + 1])
            for sub_idx, sub_tp in enumerate(sub_tps):
                if sub_tp.startswith('<e'):
                    entities.append({
                        'sent_id': sent_id,
                        'ent_id': get_attr(sub_tp, 'id'),
                        'note_id': note_id,
                        'section': section,
                        'cui': get_attr(sub_tp, 'cui'),
                        'span': sub_tps[sub_idx + 1]
                    })
    return entities


def section_split(text):
    sectioned_text = re.split(HEADER_REGEX, text, flags=re.M)
    sectioned_text = [x.lstrip() for x in sectioned_text if len(x.lstrip()) > 0]
    is_header_arr = list(map(
        lambda x: re.match(HEADER_REGEX, x, re.M) is not None and len(x) <= MAX_HEADER_LEN, sectioned_text))
    return sectioned_text, is_header_arr


def remove_empty_sections(text_pieces):
    return remove_empty(text_pieces, 'h')


def _latest_note_id(text):
    return re.findall(r'<d note_id=(\S+)', text)[-1]


def get_attr(tag, attr):
    return re.search(attr + r'=([^ ]+)', tag).group(1).strip('<>: ')


def get_ent_id(tag, attr):
    return re.search(attr + r'=([^>]+)', tag).group(1).strip('<>: ')


def is_section_match(target_section, concept, raw, treecode):
    '''
    Brief Hospital Course corresponds to the following ClarityNLP sections:
    - hospital_course
    - hospital_course_by_problem
    - hospital_course_by_location
    - hospital_course_by_system
    - emergency_department_course
    - nursery_course
    - nicu_course
    - micu_course
    - general_course
    '''
    if target_section == 'hospital_course':
        return concept in [
            'hospital_course', 'hospital_course_by_problem', 'hospital_course_by_location', 'hospital_course_by_system',
            'emergency_department_course', 'nursery_course', 'nicu_course', 'micu_course', 'general_course'
        ]
    raise Exception(f'Unrecognized target section: {target_section}')


def pretty_print_note(note, remove_sections=False):
    text_pieces = note.split('<SEP>')

    raw_pieces = []
    for tp in text_pieces:
        if tp.startswith('<h'):
            raw_pieces.append(get_attr(tp, 'raw'))
        elif tp.startswith('<p'):
            raw_pieces.append(get_attr(tp, 'name'))
        else:
            raw_pieces.append(tp)
    note = ''.join(raw_pieces)

    formatted_note = note.replace('<SEP>', '').replace('<s>', ' ').replace('</s>', '').replace('</h>', '')
    if remove_sections:
        formatted_note = re.sub(r'<h.*?>', '', formatted_note)
    return formatted_note


def remove_sections(html_str, cons_to_remove):
    text_pieces = html_str.split('<SEP>')
    filt_text_pieces = []
    open_text, close_text = text_pieces[0], text_pieces[-1]
    matching_section = False
    for idx, tp in enumerate(text_pieces[1:-1]):
        if tp.startswith('<h'):
            concept = get_attr(tp, 'con')
            matching_section = concept in cons_to_remove
        if not matching_section:
            filt_text_pieces.append(tp)

    if len(filt_text_pieces) == 0:  # don't return an empty document <d> </d>
        return None

    filt_text_pieces = [open_text] + filt_text_pieces + [close_text]
    return '<SEP>'.join(filt_text_pieces)


def filter_sections(
        html_str, target_section=None, keep_target=True, breakup_matches=False, verbose=False, section_counter=None):
    """
    if keep is True, we only keep sections = target_section
    if keep is False, we keep all sections
    """
    if breakup_matches:
        assert keep_target
    text_pieces = html_str.split('<SEP>')
    filt_text_pieces = []
    is_non_adjacent = []
    open_text, close_text = text_pieces[0], text_pieces[-1]
    matching_section = False
    last_added_idx = -100
    for idx, tp in enumerate(text_pieces[1:-1]):
        if tp.startswith('<h'):
            concept = get_attr(tp, 'con')
            raw = get_attr(tp, 'raw')
            treecode = get_attr(tp, 'tree')
            matching_section = is_section_match(target_section, concept, raw, treecode)
            if section_counter is not None:
                section_counter[concept] += 1

        if keep_target and matching_section:
            filt_text_pieces.append(tp)
            is_non_adjacent.append(last_added_idx < idx - 1)
            last_added_idx = idx
        elif not keep_target and not matching_section:
            filt_text_pieces.append(tp)

        if verbose and not keep_target and matching_section:
            print(f'Found a target section in the input: {tp}')

    if len(filt_text_pieces) == 0:  # don't return an empty document <d> </d>
        return None

    if breakup_matches:
        all_text_pieces = []
        curr_match = []
        for tp, is_start in zip(filt_text_pieces, is_non_adjacent):
            if is_start:
                if len(curr_match) > 0:
                    curr_match_str = '<SEP>'.join([open_text] + curr_match + [close_text])
                    all_text_pieces.append(curr_match_str)
                    curr_match = []
            curr_match.append(tp)
        if len(curr_match) > 0:
            curr_match_str = '<SEP>'.join([open_text] + curr_match + [close_text])
            all_text_pieces.append(curr_match_str)
        return all_text_pieces
    else:
        filt_text_pieces = [open_text] + filt_text_pieces + [close_text]
        return '<SEP>'.join(filt_text_pieces)


def remove_html_tag_chars(str):
    return re.sub(r'[<>=]+', ' ', str).strip()


def remove_note_type(html_str, remove_type='discharge_summary'):
    tps = html_str.split('<SEP>')
    filtered = []
    keep = True
    for tp in tps:
        if tp.startswith('<d'):
            keep = remove_type not in get_attr(tp, 'note_type')
        if keep:
            filtered.append(tp)
    return '<SEP>'.join(filtered)


def sectionize(html_str):
    tps = html_str.split('<SEP>')
    sections = []
    section_names = []
    curr_section, curr_section_name = None, None
    for idx, tp in enumerate(tps):
        if tp.startswith('<h'):
            curr_section = []
            curr_section_name = get_attr(tp, 'con')
        elif tp.startswith('<s'):
            curr_section.append(remove_tags_from_sent(tps[idx + 1]))
        elif tp == '</h>':
            section_names.append(curr_section_name)
            sections.append(curr_section)

    return sections, section_names


def get_sent_idxs(html_str):
    tps = html_str.split('<SEP>')
    sent_ids = []
    for idx, tp in enumerate(tps):
        if tp.startswith('<s'):
            sent_id = int(get_attr(tp, 'idx'))
            if len(sent_ids) > 0:
                assert sent_id > sent_ids[-1]
            sent_ids.append(sent_id)
    return sent_ids


def remove_tags_from_sent(text, include_headers=False, include_subheaders=False):
    text = text.replace('<SEP>', '')
    split_text, is_tag = split_by_tag(text)
    final_text = []
    for elem, tag in zip(split_text, is_tag):
        elem = elem.strip()
        if tag and elem.startswith('<h') and include_headers:
            final_text.append(' '.join(get_attr(elem, 'raw').split('_')) + ':')
        elif tag and elem.startswith('<p') and include_subheaders:
            final_text.append(' '.join(get_attr(elem, 'name').split('_')) + ':')
        elif tag or len(elem) == 0:
            continue
        else:
            final_text.append(elem)
    return ' '.join(final_text)


def safe_html(str):
    return re.sub(r'[\s_]+', '_', remove_html_tag_chars(str)).strip(string.punctuation + '_ ')


def same_text(a, b):
    return re.sub(r'\W+', '', a.lower()) == re.sub(r'\W+', '', b.lower())


def sents_from_html(html_str):
    tps = html_str.split('<SEP>')
    return [tps[idx + 1] for idx, tp in enumerate(tps) if tp.startswith('<s') and idx + 1 < len(tps)]


def sent_toks_from_html(text):
    return list(itertools.chain(*list(map(lambda x: x.split(' '), sents_from_html(text)))))


def split_by_tag(str):
    pat = re.compile(HTML_REGEX)
    text_pieces = [x for x in pat.split(str) if len(x.strip()) > 0]
    is_tag = list(map(lambda x: pat.search(x) is not None, text_pieces))
    return text_pieces, is_tag


def split_by_tag_keep_spaces(str):
    pat = re.compile(HTML_REGEX_NO_SPACE)
    text_pieces = [x for x in pat.split(str) if len(x.strip()) > 0]
    is_tag = list(map(lambda x: pat.search(x) is not None, text_pieces))
    return text_pieces, is_tag


def non_html_tok_ct(html_str):
    text_pieces = html_str.split('<SEP>')
    tok_ct = 0
    for tp in text_pieces:
        if tp.startswith('<h'):
            text_str = get_attr(tp, 'raw')
        elif tp.startswith('<p'):
            text_str = get_attr(tp, 'name')
        elif not tp.startswith('<'):
            text_str = tp
        else:
            text_str = None

        if text_str is not None:
            tok_ct += len(text_str.split(' '))
    return tok_ct
