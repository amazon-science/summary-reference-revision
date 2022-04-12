# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ujson
from glob import glob

from backoff import on_exception, expo
import boto3 as boto3
import argparse
import pandas as pd
from p_tqdm import p_uimap
from ratelimit import limits, RateLimitException
from tqdm import tqdm


from comp_med_dsum_eval.preprocess.sec_tag.section_utils import get_attr


@on_exception(expo, RateLimitException, max_tries=8)
@limits(calls=6, period=1)
def call_api(doc, func):
    try:
        result = func(Text=doc)['Entities']
    except Exception as ex:
        if 'ThrottlingException' in str(ex.args):
            raise Exception('ThrottlingException')
        else:
            print('Not a Throttling exception.  See below.')
            print(ex)
            return None

    return result


def get_prod_hera_client():
    client = boto3.client(service_name='comprehendmedical', region_name='us-west-2')
    return client


def get_ents(text, client):
    return {
        'icd': call_api(text, client.infer_icd10_cm),
        'rx': call_api(text, client.infer_rx_norm),
        'ent': call_api(text, client.detect_entities_v2)
    }


def _attach_entities(meta, link, ent_objs):
    curr_meta_idx = 0
    overflow_ct = 0
    for ent_obj in list(sorted(ent_objs, key=lambda x: x['BeginOffset'])):
        ent_obj['link'] = link
        ent_start_idx = ent_obj['BeginOffset']
        while meta[curr_meta_idx]['global_end'] < ent_start_idx:
            curr_meta_idx += 1
        curr_meta = meta[curr_meta_idx]
        ent_text = ent_obj['Text']
        chunk_text = curr_meta['text']
        trunc_idx = len(ent_text)
        ent_obj['overflow'] = False
        if curr_meta['global_end'] < ent_obj['EndOffset']:
            # print(f'Overflow of {ent_text} into {chunk_text}')
            trunc_idx -= (ent_obj['EndOffset'] - curr_meta['global_end'])
            ent_obj['overflow'] = True
            overflow_ct += 1
        assert ent_text[:trunc_idx] in chunk_text
        ent_obj['BeginOffset'] -= curr_meta['global_start']
        ent_obj['EndOffset'] -= curr_meta['global_start']
        meta[curr_meta_idx]['ents'].append(ent_obj)
    # print(f'{overflow_ct}/{len(ent_objs)} overflow')


def attach_entities(meta, ent_objs):
    _attach_entities(meta, 'icd', ent_objs['icd'])
    _attach_entities(meta, 'rx', ent_objs['rx'])
    _attach_entities(meta, 'ent', ent_objs['ent'])
    return meta


def extract_ents(html_str, client):
    tps = html_str.split('<SEP>')
    curr_note_idx = ''
    curr_sec_idx = ''
    meta = []
    full_input_str = ''
    all_ent_objs = []
    for idx, tp in enumerate(tps):
        if tp.startswith('<d'):
            curr_note_idx = int(get_attr(tp, 'idx'))
        elif tp.startswith('<h'):
            curr_sec_idx = int(get_attr(tp, 'idx'))
            sec_header = ' '.join(get_attr(tp, 'raw').split('_'))
            ent_obj = {
                'dtype': 'sec',
                'sec_idx': curr_sec_idx,
                'text': sec_header,
                'note_idx': curr_note_idx,
                'global_start': len(full_input_str) + 1,
                'global_end': len(full_input_str) + len(sec_header) + 1,
                'ents': []
            }
            meta.append(ent_obj)
            full_input_str += '\n' + sec_header + ':\n'
            assert full_input_str[ent_obj['global_start']:ent_obj['global_end']] == sec_header
        elif tp.startswith('<p'):
            para_str = ' '.join(get_attr(tp, 'name').split('_')).strip()
            ent_obj = {
                'para_idx': int(get_attr(tp, 'idx')),
                'dtype': 'para',
                'text': para_str,
                'sec_idx': curr_sec_idx,
                'note_idx': curr_note_idx,
                'global_start': len(full_input_str),
                'global_end': len(full_input_str) + len(para_str),
                'ents': []
            }
            full_input_str += para_str + ': '
            meta.append(ent_obj)
            assert full_input_str[ent_obj['global_start']:ent_obj['global_end']] == para_str
        elif tp.startswith('<s'):
            sent_str = tps[idx + 1]
            ent_obj = {
                'sent_idx': int(get_attr(tp, 'idx')),
                'text': sent_str,
                'dtype': 'sent',
                'sec_idx': curr_sec_idx,
                'note_idx': curr_note_idx,
                'global_start': len(full_input_str),
                'global_end': len(full_input_str) + len(sent_str),
                'ents': []
            }
            full_input_str += sent_str + ' '
            meta.append(ent_obj)
            assert full_input_str[ent_obj['global_start']:ent_obj['global_end']] == sent_str

        if len(full_input_str) > 1000:
            ent_extractions = get_ents(full_input_str, client)
            # adjust everything forward by current offset
            # add ent extractions to meta and reset meta
            all_ent_objs += attach_entities(meta, ent_extractions)
            meta = []
            full_input_str = ''
    if len(full_input_str) > 0:
        ent_extractions = get_ents(full_input_str, client)
        all_ent_objs += attach_entities(meta, ent_extractions)

    return all_ent_objs


def process(record, out_dir):
    client = get_prod_hera_client()
    source_ents = extract_ents(record['source'], client)
    target_ents = extract_ents(record['target'], client)
    obj = {
        'source': source_ents,
        'target': target_ents
    }

    example_id = record['example_id']
    out_fn = os.path.join(out_dir, f'{example_id}.json')
    with open(out_fn, 'w') as fd:
        ujson.dump(obj, fd)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extracting entities with Comprehend Medical from MIMIC')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=1.0, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, args.target)
    mini_str = '_mini' if args.debug else ''
    in_fn = os.path.join(input_dir, f'summary_dataset{mini_str}.csv')
    out_dir = os.path.join(input_dir, 'acm_output')

    processed_fns = glob(os.path.join(out_dir, '*.json'))
    processed_example_ids = set([x.split('/')[-1].replace('.json', '') for x in processed_fns])

    os.makedirs(out_dir, exist_ok=True)
    print('Reading in {}'.format(in_fn))
    df = pd.read_csv(in_fn)
    prev_n = len(df)
    df.dropna(inplace=True)
    df = df.sort_values(by='source', key=lambda x: x.str.len())
    n = len(df)
    print(f'{prev_n - n} rows with empty source or target')
    df = df[~df['example_id'].isin(processed_example_ids)]
    print(f'Processing {len(df)} records instead of {n} because some have already been processed')
    records = df.to_dict('records')
    print('Processing {} examples'.format(len(df)))
    if args.cpu_frac == -1:
        num_ents = list(tqdm(map(lambda record: process(record, out_dir), records)))
    else:
        num_ents = list(
            p_uimap(lambda record: process(record, out_dir), records, num_cpus=args.cpu_frac))

    print(sum(num_ents))
