# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
from pathlib import Path
import random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import torch
from transformers import AutoTokenizer

from comp_med_dsum_eval.gen_transformers.dataset import SummaryDataModule
from comp_med_dsum_eval.gen_transformers.model import TransformerSummarizer
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.ref_reviser.generate import load_reviser


def get_path_from_exp(weights_dir, experiment):
    dir = os.path.join(weights_dir, experiment)
    paths = list(Path(dir).rglob('*.ckpt'))
    if len(paths) == 0:
        raise Exception(f'No weights found in {dir}')
    elif len(paths) == 1:
        return str(paths[0])
    else:
        print('\n'.join([str(x) for x in paths]))
        raise Exception('Multiple possible weights found.  Please remove one or specify the path with --restore_path')


def run(args):
    if args.gpu_device is not None:
        gpus = [args.gpu_device]
    else:
        gpus = get_free_gpus() if torch.cuda.is_available() and not args.cpu else None
        assert gpus is None or len(gpus) > 0
        if gpus is not None and (args.debug or args.find_lr):
            gpus = [gpus[0]]
        if gpus is not None and len(gpus) > args.max_gpus:
            gpus = gpus[:args.max_gpus]
        if gpus is not None:
            gpu_str = ','.join([str(x) for x in gpus])
            print(f'Using GPUS --> {gpu_str}...')

    args.num_gpus = None if gpus is None else len(gpus)
    effective_bs = 1 if args.num_gpus is None else args.num_gpus
    assert effective_bs <= args.target_batch_size
    args.grad_accum = max(1, args.target_batch_size // effective_bs)
    gpu_num_str = 0 if args.num_gpus is None else args.num_gpus
    actual_effective_bs = args.grad_accum * gpu_num_str
    print(f'Effective BS={actual_effective_bs}. {args.grad_accum} (grad accum) x {gpu_num_str} (gpu count)')
    print('Num GPUs --> {}'.format(args.num_gpus))
    precision = 16 if args.num_gpus is not None else 32
    args.low_weight, args.mid_weight, args.high_weight = [float(x) for x in args.quality_weights.split(',')]
    if args.from_reviser_checkpoint:
        reviser = load_reviser(args.data_dir, wandb_name=args.reviser_wandb_name)
        tokenizer, bart_model = reviser['tokenizer'], reviser['bart']
        assert args.hf_model == 'facebook/bart-base'  # unless you change reviser model
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
        bart_model = None

    experiment_dir = os.path.join(args.weight_dir, args.experiment)
    os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable
    if args.control_hallucinations:
        add_tokens = ['<low-halluc>', '<mid-halluc>', '<high-halluc>']
        special_tokens_dict = {'additional_special_tokens': add_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer_out_dir = os.path.join(experiment_dir, 'tokenizer')
        os.makedirs(tokenizer_out_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_out_dir)

    model = TransformerSummarizer(args, tokenizer=tokenizer, hf_model=args.hf_model, bart_model=bart_model)

    datamodule = SummaryDataModule(
        args, tokenizer=tokenizer, max_val_num=args.max_val_num, ignore_hallucinated_ents=args.ignore_hallucinated_ents
    )

    logger = pl_loggers.WandbLogger(
        name=args.experiment,
        save_dir=experiment_dir,
        offline=args.debug or args.offline,
        project='mimic-sum',
        entity='griffinadams',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        save_last=True,
        mode='min'
    )
    # TODO - do we want to remove early stopping
    # early_stopping = EarlyStopping('val_loss', patience=5, verbose=True)
    callbacks = [checkpoint_callback]
    if not (args.no_schedule or args.debug or args.find_lr):
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    plugins = DDPPlugin(find_unused_parameters=False) if args.num_gpus is not None and args.num_gpus > 1 else None

    is_filter = args.require_admission_note or min([args.low_weight, args.mid_weight, args.high_weight]) == 0
    val_check_interval = 1.0 if is_filter else 0.25
    trainer = pl.Trainer.from_argparse_args(
        args,
        resume_from_checkpoint=args.restore_path,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        accelerator=None if args.num_gpus is None or args.num_gpus == 1 else 'ddp',
        gpus=gpus,
        default_root_dir=experiment_dir,
        gradient_clip_val=0.1,
        val_check_interval=1.0 if args.debug else val_check_interval,
        check_val_every_n_epoch=args.max_epochs if args.debug else 1,
        num_sanity_val_steps=2,
        accumulate_grad_batches=args.grad_accum,
        log_every_n_steps=2,
        max_steps=args.max_steps,
        plugins=plugins
    )

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, min_lr=1e-4, max_lr=1e-2, update_attr=True, num_training=100)
        print(lr_finder.results)
    else:
        print('Starting training...')
        trainer.fit(model, datamodule=datamodule)
        print(f'Best weights saved --> {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LongFormer/BigBird/Bart trainer.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--restore_path', default=None)
    # Train from the pre-trained reviser
    parser.add_argument('-from_reviser_checkpoint', default=False, action='store_true')
    # which reviser weights to pull iff -from_reviser_checkpoint=True
    parser.add_argument('--reviser_wandb_name', default='yay')
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--max_gpus', default=1, type=int)
    parser.add_argument('-no_schedule', default=False, action='store_true')
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-find_lr', default=False, action='store_true')
    parser.add_argument('-offline', default=False, action='store_true')
    parser.add_argument('--max_val_num', default=128, type=int)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('-require_admission_note', default=False, action='store_true')
    parser.add_argument('--version', default='original', choices=[
        'original',
        'revised_balanced',
        'revised_max_coverage',
        'revised_extractive'
    ])
    parser.add_argument('--reviser_experiment', default='yay', choices=[
        'yay', 'no_mask', 'no_neg', 'no_redress', 'no_same_sum',
    ])
    # Loss Truncation Parameters (Kang et al, 2020)
    parser.add_argument('--dropc', default=0.0, type=float)  # Fraction to drop (0 means no truncation)
    parser.add_argument('--drop_warmup_steps', default=5000, type=int)  # steps until implement truncation
    # Loss Masking Parameters (Goyal et al, 2020)
    parser.add_argument('-ignore_hallucinated_ents', default=False, action='store_true')
    # Controlled Hallucinations (Filippova 2020)
    parser.add_argument('-control_hallucinations', default=False, action='store_true')

    # Quality Filters
    # -require_admission_note = Filters training data for admission note examples
    # quality_weights = 1,1,1 -> skewed 0.5,1,2 -> only high quality train 0,0,1
    parser = TransformerSummarizer.add_model_specific_args(parser)

    args = parser.parse_args()

    if args.dropc > 0:
        assert args.drop_warmup_steps < args.max_steps  # Has no effect otherwise (No loss truncation is performed)
    args.data_dir = os.path.join(args.input_dir, args.target)
    assert os.path.exists(args.data_dir)
    args.weight_dir = os.path.join(args.data_dir, 'mimic_sum', 'weights')
    os.makedirs(args.weight_dir, exist_ok=True)
    args.results_dir = os.path.join(args.data_dir, 'mimic_sum', 'results')
    os.makedirs(args.results_dir, exist_ok=True)

    # Set same random seed for each run
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run(args)
