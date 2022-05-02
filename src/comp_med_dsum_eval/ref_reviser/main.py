# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random

import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import torch
from transformers import AutoTokenizer

from comp_med_dsum_eval.ref_reviser.dataset import ReviseDataModule
from comp_med_dsum_eval.ref_reviser.model import TransformerReviser
from comp_med_dsum_eval.gpu_utils import get_free_gpus


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PERTURB_MODEL = '/efs/griadams/dsum/weights/perturber/ent_sample/checkpoint-100000'


def extend_tokenizer(tokenizer, add_control_codes=True):
    extra_special_toks = ['<sep>']
    if add_control_codes:
        source_extract = [f'<source-extract-{i}>' for i in range(11)]
        input_extract = [f'<input-extract-{i}>' for i in range(11)]
        extra_special_toks += source_extract + input_extract
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': extra_special_toks})
    print(f'Adding {num_added_toks} special prompt tokens to tokenizer and embedding matrix')
    return extra_special_toks


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Learning to Revise Noisy Reference Sentences')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default='default', required=True)
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('-find_lr', default=False, action='store_true')
    parser.add_argument('-offline', default=False, action='store_true')
    parser.add_argument('--max_val_examples', default=512, type=int)
    parser.add_argument('--max_gpus', default=1, type=int)
    parser.add_argument('-no_schedule', default=False, action='store_true')
    parser.add_argument('--restore_path', default=None, type=str)
    parser.add_argument('--seed', default=1992, type=int)
    # Ablations for evaluation
    parser.add_argument('-remove_codes', default=False, action='store_true')
    parser.add_argument('-remove_contrast', default=False, action='store_true')
    parser.add_argument('-remove_mask', default=False, action='store_true')
    parser.add_argument('-remove_neg', default=False, action='store_true')
    parser.add_argument('-remove_redress', default=False, action='store_true')
    parser.add_argument('-remove_same_sum', default=False, action='store_true')
    parser.add_argument('--contrast_input_strategy', default='worst', choices=['worst', 'random', 'best'])

    parser = TransformerReviser.add_model_specific_args(parser)
    args = parser.parse_args()

    assert not (args.remove_contrast and args.remove_mask)

    # Set same random seed for each run
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gpus = get_free_gpus() if torch.cuda.is_available() and not args.cpu else None
    assert gpus is None or len(gpus) > 0
    if gpus is not None and (args.debug or args.find_lr):
        gpus = [gpus[0]]
    if gpus is not None and len(gpus) > args.max_gpus:
        gpus = gpus[:args.max_gpus]
    if gpus is not None:
        gpu_str = ','.join([str(x) for x in gpus])
        print(f'Using GPUs with ids --> {gpu_str}...')

    args.num_gpus = None if gpus is None else len(gpus)
    effective_bs = 1 if args.num_gpus is None else args.num_gpus
    assert effective_bs <= args.target_batch_size
    args.grad_accum = max(1, args.target_batch_size // effective_bs)
    gpu_num_str = 0 if args.num_gpus is None else args.num_gpus
    actual_effective_bs = args.grad_accum * gpu_num_str
    print(f'Effective BS={actual_effective_bs}. {args.grad_accum} (grad accum) x {gpu_num_str} (gpu count)')
    print('Num GPUs --> {}'.format(args.num_gpus))
    precision = 16 if args.num_gpus is not None else 32

    if args.debug:
        args.hf_model = 'sshleifer/bart-tiny-random'
    elif args.from_perturb_checkpoint:
        args.hf_model = PERTURB_MODEL

    data_dir = os.path.join(args.input_dir, args.target)
    args.weight_dir = os.path.join(data_dir, 'revise', 'weights')
    os.makedirs(args.weight_dir, exist_ok=True)
    mini_str = '_mini' if args.debug else ''
    data_fn = os.path.join(data_dir, 'revise', f'dataset{mini_str}.json')
    experiment_dir = os.path.join(args.weight_dir, args.experiment)
    os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable

    print(f'Loading tokenizer from {args.hf_model}')
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    extend_tokenizer(tokenizer, add_control_codes=not args.remove_codes)
    tokenizer_out_dir = os.path.join(experiment_dir, 'tokenizer')
    os.makedirs(tokenizer_out_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_out_dir)
    print(f'Instantiating TransformerReviser model...')
    model = TransformerReviser(args, tokenizer)
    datamodule = ReviseDataModule(
        data_dir, data_fn, tokenizer, debug=args.debug,
        denoise_only=args.remove_contrast, contrast_only=args.remove_mask,
        pos_only=args.remove_neg, contrast_input_strategy=args.contrast_input_strategy,
        remove_redress=args.remove_redress, remove_same_sum=args.remove_same_sum, remove_codes=args.remove_codes
    )

    logger = pl_loggers.WandbLogger(
        name=args.experiment,
        save_dir=experiment_dir,
        offline=args.debug or args.offline,
        project='ref-improve',
        entity='griffinadams',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        save_last=False,
        mode='min'
    )
    early_stopping = EarlyStopping('val_loss', patience=5, verbose=True)
    callbacks = [early_stopping, checkpoint_callback]
    if not (args.no_schedule or args.debug or args.find_lr):
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    plugins = DDPPlugin(find_unused_parameters=False) if args.num_gpus is not None and args.num_gpus > 1 else None
    trainer = pl.Trainer.from_argparse_args(
        args,
        resume_from_checkpoint=args.restore_path,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        accelerator=None if args.num_gpus is None or args.num_gpus == 1 else 'ddp',
        gpus=gpus,
        default_root_dir=experiment_dir,
        gradient_clip_val=1.0,
        val_check_interval=1.0 if args.debug else 0.1,
        check_val_every_n_epoch=10 if args.debug else 1,
        num_sanity_val_steps=2,
        accumulate_grad_batches=args.grad_accum,
        log_every_n_steps=2,
        max_steps=args.max_steps,
        plugins=plugins
    )

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(
            model, datamodule=datamodule, min_lr=1e-5, max_lr=1e-3, update_attr=True, num_training=100)
        print(lr_finder.results)
    else:
        print('Starting training...')
        trainer.fit(model, datamodule=datamodule)
        print(f'Best weights saved --> {checkpoint_callback.best_model_path}')
