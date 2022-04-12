# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import pickle
import ujson
from glob import glob

import argparse
import faiss
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
import regex as re
import sent2vec
from tqdm import tqdm

from comp_med_dsum_eval.eval.rouge import preprocess_sentence
from comp_med_dsum_eval.preprocess.entity.process_ents import add_ent_id, filter_ents


def embed_record(fn, model):
    note_id = fn.split('/')[-1].split('.')[0]
    with open(fn, 'r') as fd:
        all_ents = ujson.load(fd)

    vecs = []
    outputs = []
    num_removed = 0
    for ent_obj in all_ents:
        if ent_obj['dtype'] != 'sent':
            continue

        text = ent_obj['text']
        num_toks = len(text.split(' '))
        # Don't keep really short sentences, long sentences, or ones with no entities
        ents = filter_ents(add_ent_id(ent_obj, 'dsum'), link_col='link')
        num_ents = len(ents)
        if num_toks < 5 or num_toks > 100 or num_ents == 0:
            num_removed += 1
            continue
        annotated_text = ''
        prev_start = 0
        for ent in ents:
            start, end = ent['BeginOffset'], ent['EndOffset']
            prev_chunk = text[prev_start:start]
            prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
            annotated_text += prev_chunk.rstrip()
            category = ent['Category']
            type = ent['Type']
            ent_id = ent['ent_id']
            ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
            annotated_text += f'<e cat={category} type={type} id={ent_id}>{ent_space_pre}' + text[start:end] + '</e>'
            prev_start = end
        annotated_text += text[prev_start:]
        row = {
            'note_id': note_id,
            'num_ents': num_ents,
            'sent_idx': ent_obj['sent_idx'],
            'sec_idx': ent_obj['sec_idx'],
            'note_idx': ent_obj['note_idx'],
            'text': annotated_text
        }
        outputs.append(row)
        preprocessed_text = preprocess_sentence(text)
        vec = model.embed_sentence(preprocessed_text)[0]
        vecs.append(vec)
    return outputs, vecs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing sentence index for discharge summaries.')
    parser.add_argument('--input_dir', default='/efs/griadams/dsum')
    parser.add_argument('--cpu_frac', default=1.0, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    print('Loading BioSentVec')
    model = sent2vec.Sent2vecModel()
    model_fn = '/efs/griadams/biovec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
    model.load_model(model_fn)

    ent_dir = os.path.join(args.input_dir, 'acm_output')
    ent_pattern = ent_dir + '/*.json'
    print('Searching for entities...')
    ent_fns = glob(ent_pattern)

    if args.cpu_frac == -1:
        outputs = list(tqdm(map(lambda fn: embed_record(fn, model), ent_fns)))
    else:
        outputs = list(p_uimap(lambda fn: embed_record(fn, model), ent_fns, num_cpus=args.cpu_frac))

    output_df = pd.DataFrame(list(
        itertools.chain(*[
            x[0] for x in outputs
        ])
    ))

    flat_embeds = np.stack(list(
        itertools.chain(*[
            x[1] for x in outputs
        ])
    )).astype('float32')

    print(f'Adding {len(output_df)} sentences to the index')

    index = faiss.IndexFlatL2(700)
    index.add(flat_embeds)
    index_fn = os.path.join(args.input_dir, 'sent_index.bin')
    faiss.write_index(index, index_fn)

    embed_fn = os.path.join(args.input_dir, 'sent_embeds.pk')
    with open(embed_fn, 'wb') as fd:
        pickle.dump(flat_embeds, fd, protocol=4)

    meta_fn = os.path.join(args.input_dir, 'sent_index_meta.csv')
    output_df = output_df.assign(faiss_idx=range(len(output_df)))
    output_df.set_index('faiss_idx', drop=True, inplace=True)
    output_df.to_csv(meta_fn)
