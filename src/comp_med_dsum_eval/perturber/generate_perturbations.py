# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import regex as re

import argparse
from glob import glob
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import torch

from comp_med_dsum_eval.perturber.dataset import AlterCodeGenerateDataset, SampleGenerateDataset
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.perturber.utils import filter_df_for_eval, filter_out_done, get_chunk, set_seeds


def dump_example_outputs(noise_dir, example_outputs, example_id):
    for x in example_outputs:
        assert x['example_id'] == example_id
    out_fn = os.path.join(noise_dir, f'{example_id}.csv')
    example_outputs = pd.DataFrame(example_outputs)
    example_outputs.drop_duplicates(subset=['text'], inplace=True)
    example_outputs.to_csv(out_fn, index=False)


class CustomCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        model_inputs = self.tokenizer(
            list(itertools.chain(*[x.pop('inputs') for x in batch])),
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        return model_inputs, batch


def generate_perturbed_texts(gpu, generate_dataset, checkpoint_path, tokenizer, verbose=False):
    device = torch.device(f'cuda:{gpu}')
    num_samples = generate_dataset.samples
    print(f'Loading pre-trained model from {checkpoint_path} and putting on {device}')
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
    batch_size = 16
    data_loader = DataLoader(
        dataset=generate_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=CustomCollate(tokenizer),
        num_workers=4
    )
    example_outputs = []
    prev_example_id = None
    for inputs, batch_meta in tqdm(data_loader, total=len(data_loader)):
        max_seq_len = inputs['input_ids'].size()[1]
        kwargs = {
            'input_ids': inputs['input_ids'].to(device),
            'attention_mask': inputs['attention_mask'].to(device),
            'use_cache': True,
            'num_beams': 4,
            'min_length': 3,
            'max_length': max(128, max_seq_len + 64),
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
        }

        generated_ids = model.generate(**kwargs)
        all_generated_strs = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        all_masked_strs = tokenizer.batch_decode(inputs['input_ids'].tolist(), skip_special_tokens=False)
        for batch_idx in range(len(all_generated_strs) // num_samples):
            generated_strs = all_generated_strs[batch_idx * num_samples:(batch_idx + 1) * num_samples]
            masked_strs = all_masked_strs[batch_idx * num_samples:(batch_idx + 1) * num_samples]
            meta = batch_meta[batch_idx]
            example_id = meta['example_id']
            if prev_example_id is not None and example_id != prev_example_id:
                dump_example_outputs(noise_dir, example_outputs, prev_example_id)
                example_outputs = []
            prev_example_id = example_id

            if verbose:
                print('\n')
                print(meta['text_original'])
                print('\n')
                print('\n'.join(generated_strs))

            for perturb_idx, (text, masked_str) in enumerate(zip(generated_strs, masked_strs)):
                output = meta.copy()
                output['text'] = text
                output['perturb_idx'] = perturb_idx
                output['masked_str'] = masked_str
                example_outputs.append(output)

    dump_example_outputs(noise_dir, example_outputs, prev_example_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating Perturbations of High Quality Reference Sentences')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--wandb_name', default=None, required=True)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument('-no_ent', default=False, action='store_true')
    parser.add_argument('-only_eval', default=False, action='store_true')
    parser.add_argument('--max_n', default=99999999999, type=int)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--sample_strategy', default='resample', choices=['resample', 'alter_codes'])
    parser.add_argument('--chunksize', default=8, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)
    parser.add_argument('--gpu_device', default=None, type=int)
    # Ablations (for paper)
    parser.add_argument(
        '-no_ent_trick', default=False, action='store_true', help='Remove entity trick (as an ablation).')
    parser.add_argument('--ent_ctrl_add', default=1, type=int)

    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = args.wandb_name

    weights_dir = os.path.join(args.input_dir, 'dsum', 'weights', 'perturber', args.wandb_name)
    checkpoint_paths = glob(weights_dir + '/*')
    # Find latest checkpoint
    latest_idx = int(np.argmax([int(re.search(r'checkpoint-(\d+)', x).group(1)) for x in checkpoint_paths]))
    checkpoint_path = checkpoint_paths[latest_idx]

    # Set same random seed for each run
    set_seeds(args.seed)

    if args.debug:
        args.num_gpus = 1

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    data_dir = os.path.join(args.input_dir, args.target)
    noise_dir = os.path.join(data_dir, 'perturb', args.experiment, 'output')
    print(f'Creating {noise_dir} directory if it doesn\'t already exist')
    os.makedirs(noise_dir, exist_ok=True)

    mini_str = '_mini' if args.debug else ''
    data_fn = os.path.join(data_dir, 'high_quality', f'sents_w_related_ents{mini_str}.csv')
    print(f'Reading in data from {data_fn}...')
    data_df = pd.read_csv(data_fn)
    print(f'Loaded {len(data_df)} sentences')
    if args.only_eval:
        data_df = filter_df_for_eval(data_df, data_dir)

    if args.max_n < len(data_df):
        print(f'Randomly sampling {args.max_n} examples')
        data_df = data_df.sample(n=args.max_n, replace=False, random_state=1992)

    free_gpus = get_free_gpus()
    assert len(free_gpus) >= args.num_gpus
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device
    if args.gpu_device is not None and args.gpu_device not in free_gpus:
        print(f'Warning! Youve selected a GPU that is not available.  Putting the model on {free_gpus[0]} instead.')
        gpu = free_gpus[0]

    if args.only_new:
        data_df = filter_out_done(data_df, done_dir=noise_dir, suffix='csv')

    # Generate for a specific chunk_idx of the data, with size specified by chunksize
    if args.chunk_idx is not None:
        data_df = get_chunk(data_df, args.chunk_idx, args.chunksize, sort_first=True)

    data_df.sort_values(by=['example_id', 'sent_uid'], inplace=True)
    if args.sample_strategy == 'resample':
        dataset = SampleGenerateDataset(
            data_df, tokenizer, samples=5, no_ent=args.no_ent, ent_ctrl_add=args.ent_ctrl_add,
            no_ent_trick=args.no_ent_trick
        )
    else:
        dataset = AlterCodeGenerateDataset(data_df, tokenizer, samples=5)
    generate_perturbed_texts(gpu, dataset, checkpoint_path, tokenizer, verbose=args.debug)
