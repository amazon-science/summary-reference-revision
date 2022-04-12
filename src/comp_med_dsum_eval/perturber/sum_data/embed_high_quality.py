# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import argparse
import pandas as pd
from p_tqdm import p_uimap
import sent2vec
from tqdm import tqdm

from comp_med_dsum_eval.eval.rouge import preprocess_sentence
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent


def embed_record(record, model):
    preprocessed_text = preprocess_sentence(remove_tags_from_sent(record['target']))
    vec = model.embed_sentence(preprocessed_text)[0]
    uuid = str(record['example_id']) + '.' + str(record['sent_idx'])
    return {
        'sent_uid': uuid,
        'example_id': record['example_id'],
        'sent_idx': record['sent_idx'],
        'text': record['target'],
        'vec': vec.astype('float32')
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing sentence index for high quality reference sentences.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    mini_str = '_mini' if args.debug else ''

    print('Loading BioSentVec')
    model = sent2vec.Sent2vecModel()
    model_fn = '/efs/griadams/biovec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
    model.load_model(model_fn)

    data_dir = os.path.join(args.input_dir, args.target)
    input_dir = os.path.join(data_dir, 'revise')
    sent_fn = os.path.join(input_dir, f'sents_w_context{mini_str}.csv')
    print(f'Reading in reference sentences from {sent_fn}')
    sent_df = pd.read_csv(sent_fn)
    sent_df = sent_df[sent_df['high_quality_w_ent']]
    test_example_ids = set(pd.read_csv(os.path.join(data_dir, 'test_example_ids.csv'))['example_id'])
    sent_df = sent_df[~sent_df['example_id'].isin(test_example_ids)]

    print(f'Embedding {len(sent_df)} high quality sentences')
    sent_records = sent_df.to_dict('records')

    if args.cpu_frac == -1:
        outputs = list(tqdm(map(lambda fn: embed_record(fn, model), sent_records)))
    else:
        outputs = list(p_uimap(lambda fn: embed_record(fn, model), sent_records, num_cpus=args.cpu_frac))

    output_dir = os.path.join(args.input_dir, args.target, 'high_quality')
    os.makedirs(output_dir, exist_ok=True)
    embed_fn = os.path.join(output_dir, f'sent_embeds{mini_str}.pk')
    print(f'Saving {len(outputs)} to {embed_fn}')
    with open(embed_fn, 'wb') as fd:
        pickle.dump(outputs, fd)
