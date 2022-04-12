# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from comp_med_dsum_eval.gen_transformers.dataset import SummaryDataModule
from comp_med_dsum_eval.gen_transformers.model import TransformerSummarizer
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.ref_reviser.build_train_dataset_from_perturbed import add_to_embeds
from comp_med_dsum_eval.ref_reviser.generate import load_reviser_tokenizer


def get_path_from_exp(weights_dir, experiment, last=False):
    dir = os.path.join(weights_dir, experiment)
    paths = list(map(str, list(Path(dir).rglob('*.ckpt'))))
    if last:
        return [p for p in paths if 'last' in p][0]
    paths = [p for p in paths if 'last' not in p]
    if len(paths) == 0:
        raise Exception(f'No weights found in {dir}')
    elif len(paths) == 1:
        return str(paths[0])
    else:
        print('\n'.join([str(x) for x in paths]))
        raise Exception('Multiple possible weights found.  Please remove one or specify the path with --restore_path')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LongFormer/BigBird/Bart Generator Evaluator.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--wandb_name', default=None, required=True)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('-from_reviser_checkpoint', default=False, action='store_true')
    # only used if -from_reviser_checkpoint is true
    parser.add_argument('--reviser_wandb_name', default='yay')  # Not used unless -from_reviser_checkpoint
    parser.add_argument('--version', default='original', choices=[
        'original',  # only generate on original dataset
        # 'revised_balanced',
        # 'revised_max_coverage',
        # 'revised_extractive'
    ])
    # Controlled Hallucinations (Filippova 2020).  Only turn on if trained with -control_hallucinations
    parser.add_argument('-control_hallucinations', default=False, action='store_true')

    parser = TransformerSummarizer.add_model_specific_args(parser)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''
    if args.experiment is None:
        args.experiment = args.wandb_name
    args.data_dir = os.path.join(args.input_dir, args.target)
    assert os.path.exists(args.data_dir)
    args.weight_dir = os.path.join(args.data_dir, 'mimic_sum', 'weights')
    args.results_dir = os.path.join(args.data_dir, 'mimic_sum', 'results', args.experiment)
    os.makedirs(args.results_dir, exist_ok=True)

    free_gpus = get_free_gpus()
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device
    checkpoint_path = get_path_from_exp(args.weight_dir, args.experiment)
    if args.from_reviser_checkpoint:
        tokenizer = load_reviser_tokenizer(args.data_dir, args.reviser_wandb_name)
    else:
        try:  # We save tokenizer if we modify if from the HuggingFace default (i.e., add <halluc> attributes)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.weight_dir, args.wandb_name, 'tokenizer'))
        except:
            print(f'Loading default tokenizer for {args.hf_model}...')
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
    print(f'Loading model from {checkpoint_path}')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=checkpoint_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(gpu)

    datamodule = SummaryDataModule(args, tokenizer, for_eval_only=True)
    model.on_predict_start()
    dataloader = datamodule.test_dataloader(add_cols=['example_id'])
    outputs = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(gpu) if type(v) == torch.Tensor else v for k, v in batch.items()}
        example_id = batch['example_id'][0]
        with open(os.path.join(args.data_dir, 'embed_cache', f'{example_id}.pk'), 'rb') as fd:
            embeds = pickle.load(fd)

        # For BERT and ELECTRA
        bert_source_h = np.concatenate(
            [add_to_embeds(x, tok_col='bert_token', h_col='bert_h') for x in embeds['source']], axis=0)
        electra_source_h = np.concatenate(
            [add_to_embeds(x, tok_col='electra_tok', h_col='electra_h') for x in embeds['source']], axis=0)
        batch_stats = model.predict_step(batch, bert_source_h=bert_source_h, electra_source_h=electra_source_h)
        if type(batch_stats) == list:
            outputs += batch_stats
        else:
            outputs.append(batch_stats)

    outputs = pd.DataFrame(outputs)
    out_fn = os.path.join(args.results_dir, f'outputs{mini_str}.csv')
    print(f'Saving {len(outputs)} ROUGE scores and predictions to {out_fn}')
    outputs.to_csv(out_fn, index=False)
    num_col = outputs.select_dtypes('number')
    for col in list(num_col.columns):
        print(f'{col}: {num_col[col].dropna().mean()}')
