# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os
import pickle

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForPreTraining

from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import sents_from_html, get_sent_idxs
from comp_med_dsum_eval.perturber.evaluate import encode


def process_example(
        out_dir, record, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device, top_n_layers=4):
    example_id = record['example_id']
    outputs = {'source': [], 'target': [], 'example_id': example_id}

    target_sents = sents_from_html(record['target'])
    target_sent_idxs = get_sent_idxs(record['target'])

    source_sents = sents_from_html(record['source'])
    source_sent_idxs = get_sent_idxs(record['source'])

    texts = source_sents + target_sents
    idxs = source_sent_idxs + target_sent_idxs
    dtypes = ['source'] * len(source_sents) + ['target'] * len(target_sents)
    electra_tokens = list(map(electra_tokenizer.tokenize, texts))
    electra_outputs, electra_seq_lens = encode(
        texts, electra_model, electra_tokenizer, device=device, max_length=256, top_n_layers=top_n_layers)

    bert_tokens = list(map(clinbert_tokenizer.tokenize, texts))
    bert_outputs, bert_seq_lens = encode(
        texts, clinbert_model, clinbert_tokenizer, device, max_length=128, top_n_layers=top_n_layers)

    for i in range(len(texts)):
        electra_h = electra_outputs['hidden_states'][i, :electra_seq_lens[i]].numpy()
        logits = electra_outputs['logits'][i, :electra_seq_lens[i]].numpy()
        bert_h = bert_outputs['hidden_states'][i, :bert_seq_lens[i]].numpy()
        row = {
            'sent_idx': idxs[i], 'electra_logits': logits, 'electra_h': electra_h, 'electra_tok': electra_tokens[i],
            'bert_token': bert_tokens[i], 'bert_h': bert_h
        }
        outputs[dtypes[i]].append(row)

    out_fn = os.path.join(out_dir, f'{example_id}.pk')
    with open(out_fn, 'wb') as fd:
        pickle.dump(outputs, fd)


def process_chunk(out_dir, records, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device):
    records = list(sorted(records, key=lambda x: len(x['source'])))
    electra_model = electra_model.to(device)
    clinbert_model = clinbert_model.to(device)
    for record in tqdm(records, total=len(records)):
        process_example(out_dir, record, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate BERT and Electra embeddings one time for future use for evaluation')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument(
        '--hf_model', default='sultan/BioM-ELECTRA-Large-Discriminator',
        choices=[
            'kamalkraj/bioelectra-base-discriminator-pubmed',
            'sultan/BioM-ELECTRA-Large-Discriminator'
        ])
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('--num_gpus', default=6, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    if args.debug:
        args.cpu_frac = -1
        args.num_gpus = 1

    data_dir = os.path.join(args.input_dir, args.target)
    config = AutoConfig.from_pretrained(args.hf_model)

    electra_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    electra_model = AutoModelForPreTraining.from_pretrained(args.hf_model).eval()

    clinbert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    clinbert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').eval()

    free_gpus = get_free_gpus()

    data_fn = os.path.join(data_dir, f'summary_dataset_rouge_annotated{mini_str}.csv')
    print(f'Reading in data from {data_fn}...')
    df = pd.read_csv(data_fn)
    # TODO Remove
    test_fn = os.path.join(data_dir, 'mimic_sum', 'results', 'long_revised_balanced', 'outputs.csv')
    test_example_ids = set(pd.read_csv(test_fn)['example_id'])
    df = df[df['example_id'].isin(test_example_ids)]

    out_dir = os.path.join(data_dir, 'embed_cache')
    os.makedirs(out_dir, exist_ok=True)

    if args.only_new:
        done_fns = glob(out_dir + '/*.pk')
        done_example_ids = set([x.split('/')[-1].replace('.pk', '') for x in done_fns])
        print(f'Choosing not to recompute embeddings for the {len(done_example_ids)} already completed examples')
        df = df[~df['example_id'].isin(done_example_ids)]
    records = df.to_dict('records')

    if args.num_gpus == 1:
        process_chunk(
            out_dir, records, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, free_gpus[0])
    else:
        record_chunks = np.array_split(records, args.num_gpus)
        free_gpus = free_gpus[:len(record_chunks)]
        list(p_uimap(lambda x: process_chunk(
            out_dir, x[0], electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device=x[1]),
                     list(zip(record_chunks, free_gpus)), num_cpus=len(record_chunks)))
