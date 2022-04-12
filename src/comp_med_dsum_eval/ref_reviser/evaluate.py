# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os
import pickle
import regex as re

import argparse
import numpy as np
import pandas as pd
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from comp_med_dsum_eval.ref_reviser.build_train_dataset_from_perturbed import add_to_embeds
from comp_med_dsum_eval.perturber.evaluate import encode, bertscore_post_filt
from comp_med_dsum_eval.gen_transformers.model import h_no_special


def process_example(data_dir, out_dir, fn, clinbert_model, clinbert_tokenizer, device, uid_filter=None):
    example_id = fn.split('/')[-1].replace('.csv', '')
    df = pd.read_csv(fn)
    with open(os.path.join(data_dir, 'embed_cache', f'{example_id}.pk'), 'rb') as fd:
        embeds = pickle.load(fd)
    bert_source_h = [(x['sent_idx'], add_to_embeds(x, tok_col='bert_token', h_col='bert_h')) for x in embeds['source']]
    predictions = df['prediction'].tolist()
    bert_tokens = list(map(clinbert_tokenizer.tokenize, predictions))
    bert_outputs, bert_seq_lens = encode(
        predictions, clinbert_model, clinbert_tokenizer, device, max_length=128, top_n_layers=4
    )
    # full_bert_source_h = np.concatenate([x[1] for x in bert_source_h], axis=0)
    new_records = []
    for i, record in enumerate(df.to_dict('records')):
        pred_bert_h = h_no_special(bert_tokens[i], bert_outputs['hidden_states'][i], bert_seq_lens[i])
        sent_idxs = list(map(int, re.findall(r'idx=(\d+)', record['context'])))
        context_bert_source_h = np.concatenate([bert_source_h[used_sent_idx][1] for used_sent_idx in sent_idxs], axis=0)
        bert_con_prec, bert_con_cov, bert_con_f1 = bertscore_post_filt(pred_bert_h, context_bert_source_h)
        # bert_src_prec, bert_src_cov, bert_src_f1 = bertscore_post_filt(pred_bert_h, full_bert_source_h)
        new_records.append({
            'example_id': record['example_id'],
            'target_sent_idx': record['target_sent_idx'],
            'input_extract_code': record['input_extract_code'],
            'source_extract_code': record['source_extract_code'],
            'prediction': record['prediction'],
            'bert_bs_con_cov': bert_con_cov,
            'bert_bs_con_prec': bert_con_prec,
            'bert_bs_con_f1': bert_con_f1,
            # 'bert_bs_src_cov': bert_src_cov,
            # 'bert_bs_src_prec': bert_src_prec,
            # 'bert_bs_src_f1': bert_src_f1,
        })
    out_df = pd.DataFrame(new_records)
    out_fn = os.path.join(out_dir, f'{example_id}.csv')
    out_df.to_csv(out_fn, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluating.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default='yay')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('-only_eval', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--num_chunks', default=8, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)

    args = parser.parse_args()
    if args.debug:
        args.cpu_frac = -1
        args.num_gpus = 1

    data_dir = os.path.join(args.input_dir, args.target)
    revise_dir = os.path.join(data_dir, 'revise', 'output', args.experiment)
    out_dir = os.path.join(data_dir, 'revise', 'eval', args.experiment)
    os.makedirs(out_dir, exist_ok=True)
    fns = glob(os.path.join(revise_dir, '*.csv'))
    uid_filter = None
    if args.only_eval:
        # Only run evaluation code for the 1,000 set aside for evaluation (just faster)
        uid_filter = set(pd.read_csv(os.path.join(data_dir, 'high_quality', 'eval_examples.csv'))['uid'])

    if args.only_new:
        example_ids = [x.split('/')[-1].replace('.csv', '') for x in fns]
        meta_pattern = out_dir + '/*.csv'
        meta_ent_fns = glob(meta_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in meta_ent_fns])
        print(f'Choosing not to re-evaluate for the {len(done_example_ids)} already completed examples')
        fns = [fn for fn, example_id in zip(fns, example_ids) if example_id not in done_example_ids]

    print(f'Found {len(fns)} files')
    if args.chunk_idx is not None:
        fn_chunks = np.array_split(fns, args.num_chunks)
        fns = fn_chunks[args.chunk_idx]
        print(f'Processing chunk {args.chunk_idx + 1}/{args.num_chunks} of size={len(fns)}')

    clinbert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    clinbert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    free_gpus = get_free_gpus()
    device = free_gpus[0] if args.gpu_device is None else args.gpu_device
    assert device in free_gpus
    clinbert_model = clinbert_model.to(device)
    for fn in tqdm(fns, total=len(fns)):
        process_example(data_dir, out_dir, fn, clinbert_model, clinbert_tokenizer, device, uid_filter=uid_filter)
