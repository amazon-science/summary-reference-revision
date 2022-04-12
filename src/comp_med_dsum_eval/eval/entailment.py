# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from comp_med_dsum_eval.gpu_utils import get_free_gpus


def load_entail_scorer(hf_model='razent/SciFive-large-Pubmed_PMC-MedNLI', device=0):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_model).eval().to(device)
    return {
        'tokenizer': tokenizer,
        'model': model
    }


def compute_entailments(example_id, pred_sents_w_context, entail_scorer, device=0):
    entail_inputs = []
    for record in pred_sents_w_context.to_dict('records'):
        predicted_sent = record['prediction']
        context = ' '.join(record['context'].split('<s>'))
        entail_inputs.append(f'mednli: sentence1: {context} sentence2: {predicted_sent}')
    encoding = entail_scorer['tokenizer'](
        entail_inputs, padding='max_length', truncation=True, max_length=256, return_tensors='pt'
    )
    input_ids, attention_masks = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = entail_scorer['model'].generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=8,
            early_stopping=True
        )

    entail_labels = list(map(
        lambda x: entail_scorer['tokenizer'].decode(
            x, skip_special_tokens=True, clean_up_tokenization_spaces=True),
        outputs)
    )
    label_counts = Counter(entail_labels)
    return {
        'example_id': example_id,
        'entailment': label_counts.get('entailment', 0),
        'neutral': label_counts.get('neutral', 0),
        'contradiction': label_counts.get('contradiction', 0),
        'frac_entailed': label_counts.get('entailment', 0) / len(entail_labels)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to retrieve contexts for model predictions')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiments', default='longformer_16384_full')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--gpu_device', default=None, type=int)

    args = parser.parse_args()

    free_gpus = get_free_gpus()
    args.gpu_device = free_gpus[0] if args.gpu_device is None else args.gpu_device
    for experiment in args.experiments.split(','):
        data_dir = os.path.join(args.input_dir, args.target)
        experiment_dir = os.path.join(data_dir, 'mimic_sum', 'results', experiment)
        context_fn = os.path.join(experiment_dir, 'contexts.csv')
        gen_df = pd.read_csv(context_fn)
        example2contexts = dict(tuple(gen_df.groupby('example_id')))
        ex_ids = list(example2contexts.keys())
        entail_scorer = load_entail_scorer(device=args.gpu_device)
        outputs = pd.DataFrame(list(tqdm(map(lambda example_id: compute_entailments(
            example_id,
            example2contexts[example_id],
            entail_scorer=entail_scorer,
            device=args.gpu_device
        ), ex_ids), total=len(ex_ids))))
        out_fn = os.path.join(experiment_dir, 'entailment_scores.csv')
        print(f'Saving {len(outputs)} entailment scores for {len(gen_df)} predictions to {out_fn}')
        outputs.to_csv(out_fn, index=False)
        avg_frac_entailed = outputs['frac_entailed'].mean()
        print(f'Fraction entailed for {experiment}: {avg_frac_entailed}')
