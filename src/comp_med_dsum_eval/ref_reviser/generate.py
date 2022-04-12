# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import regex as re
import ujson

import argparse
from glob import glob
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForPreTraining
from tqdm import tqdm
import torch

from comp_med_dsum_eval.ref_reviser.dataset import GenerateDataset, tokenize
from comp_med_dsum_eval.ref_reviser.model import TransformerReviser, remove_prefix_and_tags
from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.perturber.evaluate import plausibility, encode


ELECTRA_HF_MODEL = 'sultan/BioM-ELECTRA-Large-Discriminator'


def load_reviser_tokenizer(data_dir, wandb_name='yay'):
    weights_dir = os.path.join(data_dir, 'revise', 'weights', wandb_name)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(weights_dir, 'tokenizer'))
    return tokenizer


def load_reviser(data_dir, wandb_name='yay', device=None):
    weights_dir = os.path.join(data_dir, 'revise', 'weights', wandb_name)
    checkpoint_paths = glob(weights_dir + '/ref-improve/*/*/*.ckpt', recursive=True)
    latest_ckpt = int(np.argmax([extract_priority(p) for p in checkpoint_paths]))
    checkpoint_path = checkpoint_paths[latest_ckpt]
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(weights_dir, 'tokenizer'))
    model = TransformerReviser.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer, strict=False)
    if device is not None:
        model = model.to(device)
    return {
        'bart': model.bart,
        'tokenizer': tokenizer
    }


def dump_example_outputs(revise_dir, example_outputs, example_id):
    for x in example_outputs:
        assert x['example_id'] == example_id
    out_fn = os.path.join(revise_dir, f'{example_id}.csv')
    example_outputs = pd.DataFrame(example_outputs)
    example_outputs.to_csv(out_fn, index=False)


def generate_revised_texts(gpu, generate_dataset, path, tokenizer, electra_model, electra_tokenizer, verbose=False):
    device = torch.device(f'cuda:{gpu}')
    print(f'Loading pre-trained model from {path} and putting on {device}')
    model = TransformerReviser.load_from_checkpoint(path, tokenizer=tokenizer, strict=False).to(device)
    model.on_validation_start()
    electra_model = electra_model.to(device)
    outputs = []
    all_stats = []
    prev_example_id = None
    for example, str_inputs, meta in tqdm(generate_dataset, total=len(generate_dataset)):
        example_id = meta['example_id']
        if prev_example_id is not None and example_id != prev_example_id:
            dump_example_outputs(revise_dir, outputs, prev_example_id)
            outputs = []
        prev_example_id = example_id

        max_seq_len = example['input_ids'].size()[1]
        kwargs = {
            'input_ids': example['input_ids'].to(device),
            'attention_mask': example['attention_mask'].to(device),
            'use_cache': True,
            'num_beams': 4,
            'min_length': 5,
            'max_length': max(128, max_seq_len + 64),
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
        }

        generated_ids = model.generate(**kwargs)
        generated_strs = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        generated_strs_no_dup = []
        codes_no_dup = []
        seen_strs = set()
        for text, codes in zip(generated_strs, str_inputs['meta_codes']):
            if text.lower() not in seen_strs:
                seen_strs.add(text.lower())
                generated_strs_no_dup.append(text)
                codes_no_dup.append(codes)

        n = len(generated_strs_no_dup)
        target_sent = meta['target_sent']
        target_toks = tokenize(target_sent)
        context = str_inputs['context'].replace('</s>', ' ')
        source_toks = tokenize(context)

        all_texts = generated_strs_no_dup + [target_sent, context]
        all_texts_cleaned = list(map(remove_prefix_and_tags, all_texts))
        bert_outputs, bert_seq_lens = encode(
            all_texts_cleaned, model.clinbert_model, model.clinbert_tokenizer, model.device, max_length=128)

        electra_tokens = list(map(electra_tokenizer.tokenize, generated_strs))
        electra_outputs, electra_seq_lens = encode(
            generated_strs, electra_model, electra_tokenizer, device=device, max_length=256, top_n_layers=4)

        for gen_idx in range(n):
            gen_str = generated_strs_no_dup[gen_idx]
            gen_toks = tokenize(gen_str)
            source_extractive_frags = parse_extractive_fragments(source_toks, gen_toks, remove_stop=True)
            source_extractive_frags = {'source_' + k: v for k, v in source_extractive_frags.items()}
            input_extractive_frags = parse_extractive_fragments(target_toks, gen_toks, remove_stop=True)
            input_extractive_frags = {'target_' + k: v for k, v in input_extractive_frags.items()}

            rel_idxs = [gen_idx, len(all_texts_cleaned) - 2, len(all_texts_cleaned) - 1]
            rel_bert_h = bert_outputs['hidden_states'][rel_idxs]
            rel_texts_cleaned = [all_texts_cleaned[rel_idx] for rel_idx in rel_idxs]
            metric_versions = ['generated', 'perturb', 'context']
            rel_seq_lens = [bert_seq_lens[rel_idx] for rel_idx in rel_idxs]
            stats = model.overlap_metrics(
                rel_texts_cleaned, metric_versions, bert_h=rel_bert_h, bert_seq_lens=rel_seq_lens)
            stats = {k: v for k, v in stats.items() if v is not None}
            stats.update(codes_no_dup[gen_idx])
            stats.update(source_extractive_frags)
            stats.update(input_extractive_frags)

            fake_metrics = plausibility(electra_tokens[gen_idx], electra_outputs['logits'][gen_idx, 1:-1])
            stats.update(fake_metrics)

            revise_output = meta.copy()
            revise_output.update(codes_no_dup[gen_idx])
            revise_output.update({
                'prediction': gen_str,
                'context': str_inputs['context'],
            })
            revise_output.update(stats)

            if verbose:
                print('\n')
                print(remove_prefix_and_tags(str_inputs['context']))
                print(remove_prefix_and_tags(str_inputs['target']))
                # print(str(improve_code) + ': ' + gen_str)
                print('\n')

            # Add to the lists
            outputs.append(revise_output)

            all_stats.append(stats)

    dump_example_outputs(revise_dir, outputs, prev_example_id)
    return all_stats


def extract_priority(path):
    epoch = int(re.search(r'epoch=(\d+)', path).group(1))
    steps = int(re.search(r'step=(\d+)', path).group(1))
    return epoch * 999999 + steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Revise potentially flawed reference sentences')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--wandb_name', default='yay')
    parser.add_argument('--experiment', default=None)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-max_input_length', default=1024, type=int)
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('--chunksize', default=8, type=int)
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)
    parser.add_argument('-only_eval', default=False, action='store_true')

    # Post-Hoc Editing Use Case
    parser.add_argument('-load_model_generated', default=False, action='store_true')
    parser.add_argument('--summary_experiment', default='longformer_16384_full')

    args = parser.parse_args()

    # revise / weights
    data_dir = os.path.join(args.input_dir, args.target)

    mini_str = '_mini' if args.debug else ''
    if args.load_model_generated:
        in_fn = os.path.join(data_dir, 'mimic_sum', 'results', args.summary_experiment, 'contexts.csv')
        if not os.path.exists(in_fn):
            print(f'First run python ../gen_transformers/retrieve_contexts.py --experiment {args.summary_experiment}')
        print(f'Loading model-generated summary sentence-context pairs from {in_fn}')
        examples_df = pd.read_csv(in_fn)
        # Convert it to list of objects of form: {
        # 'target_sent': {prediction},
        # 'source': {'sent_idxs': [aligned source sent idxs], 'source_sents': [aligned source sentences]},
        # 'example_id': {example_id}
        # }
        examples = []
        for row in examples_df.to_dict('records'):
            source_sents = row['context'].split('<s>')
            source = {
                'sents': source_sents,
                'sent_idxs': [-1] * len(source_sents)  # Not recorded but also not needed
            }
            examples.append({
                'example_id': row['example_id'], 'target_sent': row['prediction'], 'source': source,
                'predict_sent_idx': row['predicted_sent_idx']
            })
        if args.experiment is None:
            args.experiment = args.summary_experiment
    else:
        eval_str = '_eval' if args.only_eval else ''
        data_fn = os.path.join(data_dir, 'revise', f'low_quality_w_context{mini_str}{eval_str}.json')
        print(f'Loading low quality reference sentences dataset from {data_fn}')
        with open(data_fn, 'r') as fd:
            examples = ujson.load(fd)
        examples = list(sorted(examples, key=lambda x: x['example_id']))
        if args.experiment is None:
            args.experiment = args.wandb_name
    revise_dir = os.path.join(data_dir, 'revise', 'output', args.experiment)
    print(f'Creating {revise_dir} directory if it doesn\'t already exist')
    os.makedirs(revise_dir, exist_ok=True)

    free_gpus = get_free_gpus()
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device
    if args.gpu_device is not None and args.gpu_device not in free_gpus:
        print(f'Warning! Youve selected a GPU that is not available.  Putting the model on {free_gpus[0]} instead.')
        gpu = free_gpus[0]

    # Generate for a specific chunk_idx of the data, with size specified by chunksize
    if args.chunk_idx is not None:
        example_ids = list(np.sort(list(set([x['example_id'] for x in examples]))))
        chunks = np.array_split(example_ids, args.chunksize)
        example_id_set = set(chunks[args.chunk_idx])
        examples = [example for example in examples if example['example_id'] in example_id_set]

    if args.only_new:
        example_ids = set([x['example_id'] for x in examples])
        csv_pattern = revise_dir + '/*.csv'
        ent_fns = glob(csv_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in ent_fns])
        print(f'Choosing not to re extract entities for the {len(done_example_ids)} already completed examples')
        prev_n = len(examples)
        examples = [example for example in examples if example['example_id'] not in done_example_ids]
        n = len(examples)
        print(f'Shrunk number to be processed from {prev_n} to {n}')

    weights_dir = os.path.join(data_dir, 'revise', 'weights', args.wandb_name)
    checkpoint_paths = glob(weights_dir + '/ref-improve/*/*/*.ckpt', recursive=True)
    latest_ckpt = int(np.argmax([extract_priority(p) for p in checkpoint_paths]))
    checkpoint_path = checkpoint_paths[latest_ckpt]
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(weights_dir, 'tokenizer'))

    print(f'Loading ELECTRA model from HF -> {ELECTRA_HF_MODEL}')
    electra_tokenizer = AutoTokenizer.from_pretrained(ELECTRA_HF_MODEL)
    electra_model = AutoModelForPreTraining.from_pretrained(ELECTRA_HF_MODEL)

    dataset = GenerateDataset(examples, tokenizer)
    stats = generate_revised_texts(gpu, dataset, checkpoint_path, tokenizer, electra_model, electra_tokenizer)
    stats = pd.DataFrame(stats)
    stats = stats.select_dtypes('number')
    print(f'Showing results for {args.experiment}...')
    for col in list(stats.columns):
        print(f'{col}: {stats[col].dropna().mean()}')
