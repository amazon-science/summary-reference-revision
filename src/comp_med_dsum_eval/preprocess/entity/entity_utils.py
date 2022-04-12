# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import regex as re

from comp_med_dsum_eval.preprocess.constants import HTML_REGEX_NO_SPACE
from comp_med_dsum_eval.preprocess.entity.process_ents import filter_ents
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import get_attr


ENT_ORDER = ['dx', 'procedure', 'treatment', 'test', 'med']
ENT_TYPE_MAP = {
    'DX_NAME': 'dx',
    'PROCEDURE_NAME': 'procedure',
    'TREATMENT_NAME': 'treatment',
    'TEST_NAME': 'test',
    'BRAND_NAME': 'med',
    'GENERIC_NAME': 'med'
}


def annotate_sent_w_ents(ent_obj):
    annotated = ''
    prev_start = 0
    for ent in filter_ents(ent_obj['ents']):
        start, end = ent['BeginOffset'], ent['EndOffset']
        prev_chunk = ent_obj['text'][prev_start:start]
        prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
        annotated += prev_chunk.rstrip()
        category = ent['Category']
        type = ent['Type']
        ent_id = ent['ent_id']
        ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
        annotated += f'<e cat={category} type={type} id={ent_id}>{ent_space_pre}' + ent_obj['text'][start:end] + '</e>'
        prev_start = end
    annotated += ent_obj['text'][prev_start:]
    return annotated


def annotate_sent_w_ents_add_halluc(ent_obj, matched_ent_ids):
    annotated = ''
    prev_start = 0
    for ent in filter_ents(ent_obj['ents']):
        start, end = ent['BeginOffset'], ent['EndOffset']
        prev_chunk = ent_obj['text'][prev_start:start]
        prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
        annotated += prev_chunk.rstrip()
        category = ent['Category']
        type = ent['Type']
        ent_id = ent['ent_id']
        ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
        halluc = 0 if ent_id in matched_ent_ids else 1
        annotated += f'<e cat={category} type={type} id={ent_id} halluc={halluc}>{ent_space_pre}'\
                     + ent_obj['text'][start:end] + '</e>'
        prev_start = end
    annotated += ent_obj['text'][prev_start:]
    return annotated


def extract_ent_ids(html_str):
    tps = re.split(HTML_REGEX_NO_SPACE, html_str)
    ent_ids = []
    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<e'):
            ent_ids.append(get_attr(tp, 'id'))
    return ent_ids


def extract_ents(html_str, ent_idxs_to_remove):
    tps = re.split(HTML_REGEX_NO_SPACE, html_str)
    removed_ents = []
    removed_ent_types = []
    all_ents = []
    target = ''
    text_perturbed = ''
    ent_idx = -1
    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<e') or tp == '</e>':
            continue
        elif tp_idx > 0 and tps[tp_idx - 1].startswith('<e'):
            target += tp
            ent_idx += 1
            all_ents.append(tp.strip())
            if ent_idx in ent_idxs_to_remove:
                ent_type = ENT_TYPE_MAP[get_attr(tps[tp_idx - 1], 'type')]
                removed_ents.append(tp.strip())
                removed_ent_types.append(ent_type)
            else:
                text_perturbed += tp
        else:
            target += tp
            text_perturbed += tp
    return target, text_perturbed, removed_ents, removed_ent_types, all_ents
