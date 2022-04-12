# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os

import argparse
import numpy as np
import pandas as pd
import regex as re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForNextSentencePrediction

from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent, sents_from_html, get_sent_idxs


def nsp(s1, s2, model, tokenizer, device):
    encoding = tokenizer(s1, s2, return_tensors='pt', max_length=128, truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    dummy_label = torch.LongTensor([1]).to(device)
    with torch.no_grad():
        outputs = model(**encoding, labels=dummy_label)
    logits = outputs.logits
    prob = torch.softmax(logits, dim=1)
    return float(prob[0, 0])


def process_example(
        example_id, out_dir, target, fn, model, tokenizer, device, uid_filter=None, include_reference_nsp=True):
    examples = pd.read_csv(fn)
    target_sent_idxs = get_sent_idxs(target)
    target_sents = sents_from_html(target)
    records = list(sorted(examples.to_dict('records'), key=lambda x: (x['sent_idx'], x['perturb_idx'])))
    idx2text = {}
    for sent_idx, sent in zip(target_sent_idxs, target_sents):
        idx2text[sent_idx] = remove_tags_from_sent(sent)

    prev_texts = []
    nsp_scores = []
    eval_outputs = []
    sent_nsp_scores = {}
    for i, record in enumerate(records):
        sent_idx = record['sent_idx']
        uid = f'{example_id}.{sent_idx}'
        if uid_filter is not None and uid not in uid_filter:
            continue
        if sent_idx == 0:
            continue
        prev_sent = idx2text[sent_idx - 1]
        perturb_idx = record['perturb_idx']
        sent_key = str(sent_idx)
        if include_reference_nsp and(i == 0 or str(sent_idx) != str(records[i - 1]['sent_idx'])):
            nsp_score = nsp(prev_sent, remove_tags_from_sent(record['text_original']), model, tokenizer, device=device)
            eval_outputs.append(
                {'example_id': example_id, 'key': sent_key, 'sent_idx': sent_idx, 'version': 'original',
                 'perturb_idx': None, 'nsp': nsp_score}
            )
            sent_nsp_scores[sent_idx] = nsp_score

        if perturb_idx > 0:
            continue

        perturb_key = f'{sent_idx}_{perturb_idx}'
        nsp_score = nsp(prev_sent, remove_tags_from_sent(record['text']), model, tokenizer, device=device)
        nsp_scores.append(nsp_score)
        prev_texts.append(prev_sent)

        try:
            ent_add = int(re.search(r'<ent-add-(\d+)>', record['masked_str']).group(1)),
            ent_remove = int(re.search(r'<ent-remove-(\d+)>', record['masked_str']).group(1))
            span_remove = int(re.search(r'<span-remove-(\d+)>', record['masked_str']).group(1))
            shuffle = int(re.search(r'<shuffle-(\d+)>', record['masked_str']).group(1))
        except:
            span_remove = ent_add = ent_remove = shuffle = None

        row = {
            'example_id': example_id, 'key': perturb_key, 'sent_idx': sent_idx, 'version': 'perturbed',
            'perturb_idx': perturb_idx, 'nsp': nsp_score, 'nsp_delta': nsp_score - sent_nsp_scores[sent_idx],
            'span_remove': span_remove, 'ent_add': ent_add, 'ent_remove': ent_remove, 'shuffle_orderliness': shuffle,
        }
        eval_outputs.append(row)

    out_fn = os.path.join(out_dir, f'{example_id}.csv')
    if len(eval_outputs) > 0:
        out_df = pd.DataFrame(eval_outputs)
        out_df.to_csv(out_fn, index=False)
    return nsp_scores


def process_chunk(out_dir, ex2target, fns, model, tokenizer, device, uid_filter=None):
    model = model.to(device)
    nsp_scores = []
    for fn in tqdm(fns, total=len(fns)):
        example_id = fn.split('/')[-1].replace('.csv', '')
        nsp_scores += process_example(example_id, out_dir, ex2target[example_id], fn, model, tokenizer, device, uid_filter=uid_filter)
    print(np.mean(nsp_scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate perturb outputs for coherence')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--gpu_device', default=0, type=int)
    parser.add_argument('-only_eval', default=False, action='store_true')
    parser.add_argument(
        '--hf_model', default='emilyalsentzer/Bio_ClinicalBERT',
        choices=['emilyalsentzer/Bio_ClinicalBERT']
    )

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = BertForNextSentencePrediction.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    free_gpus = get_free_gpus()

    noise_dir = os.path.join(data_dir, 'perturb', args.experiment, 'output')
    out_dir = os.path.join(data_dir, 'perturb', args.experiment, 'nsp')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Reading in files from {noise_dir}')
    fns = glob(noise_dir + '/*.csv')
    print(f'Found {len(fns)} examples to process...')

    data_fn = os.path.join(data_dir, 'summary_dataset.csv')
    print(f'Loading in summary dataset from {data_fn}')
    summary_dataset = pd.read_csv(data_fn)
    ex2target = dict(zip(summary_dataset['example_id'], summary_dataset['target']))

    uid_filter = None
    if args.only_eval:
        # Only run evaluation code for the 1,000 set aside for evalution (just faster)
        uid_filter = set(pd.read_csv(os.path.join(data_dir, 'high_quality', 'eval_examples.csv'))['uid'])

    gpu = args.gpu_device if args.gpu_device is not None else free_gpus[0]
    process_chunk(out_dir, ex2target, fns, model, tokenizer, device=gpu, uid_filter=uid_filter)
