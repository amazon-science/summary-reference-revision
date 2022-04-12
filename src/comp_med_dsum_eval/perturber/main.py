# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
from datasets import load_metric
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
)

from comp_med_dsum_eval.perturber.dataset import ENT_TYPE_MAP, PerturbDataset
from comp_med_dsum_eval.gpu_utils import get_free_gpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Learning to perturb entities')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default='default', required=True)
    parser.add_argument('--target', default='dsum')
    parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
        'facebook/bart-base',
        'facebook/bart-large'
    ])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-tune', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--max_val_examples', default=1024, type=int)
    parser.add_argument('-no_ent', default=False, action='store_true')
    parser.add_argument('--seed', default=1956, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--gpu_no', default=None, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)

    args = parser.parse_args()

    free_gpus = get_free_gpus()
    if args.gpu_no is None:
        free_gpu = free_gpus[:min(len(free_gpus), args.num_gpus)]
    else:
        free_gpu = args.gpu_no
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in free_gpu])

    if args.debug:
        args.hf_model = 'sshleifer/bart-tiny-random'

    model = BartForConditionalGeneration.from_pretrained(args.hf_model)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    ent_remove_toks = [f'<ent-remove-{i}>' for i in range(11)]
    ent_add_toks = [f'<ent-add-{i}>' for i in range(11)]
    span_remove_toks = [f'<span-remove-{i}>' for i in range(11)]
    orderliness_toks = [f'<shuffle-{i}>' for i in range(11)]
    sep_toks = ['<ent-sep>', '<sep>']
    ent_special_toks = []
    ent_types = list(set(list(ENT_TYPE_MAP.values())))
    for ent_type in ent_types:
        ent_special_toks.append(f'<{ent_type}>')
        ent_special_toks.append(f'</{ent_type}>')
    add_tokens = sep_toks + ent_remove_toks + ent_add_toks + span_remove_toks + orderliness_toks + ent_special_toks
    special_tokens_dict = {'additional_special_tokens': add_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f'Adding {num_added_toks} special prompt tokens to tokenizer and embedding matrix')
    model.resize_token_embeddings(len(tokenizer))

    data_dir = os.path.join(args.input_dir, args.target)
    mini_str = '_mini' if args.debug else ''
    data_fn = os.path.join(data_dir, f'sent_index_meta_w_related_ents{mini_str}.csv')
    print(f'Reading in data from {data_fn}...')
    data_df = pd.read_csv(data_fn)

    train_note_ids = set(pd.read_csv(os.path.join(data_dir, 'train_note_ids.csv'))['note_id'])
    val_note_ids = set(pd.read_csv(os.path.join(data_dir, 'validation_note_ids.csv'))['note_id'])
    train_df = data_df[data_df['note_id'].isin(train_note_ids)]
    val_df = data_df[data_df['note_id'].isin(val_note_ids)]
    if len(val_df) > args.max_val_examples:
        val_df = val_df.sample(n=args.max_val_examples, replace=False, random_state=args.seed)

    train_dataset = PerturbDataset(train_df, tokenizer, no_ent=args.no_ent)
    val_dataset = PerturbDataset(val_df, tokenizer, no_ent=args.no_ent)

    weights_dir = os.path.join(data_dir, 'weights', 'perturber', args.experiment)
    os.makedirs(weights_dir, exist_ok=True)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    training_args = Seq2SeqTrainingArguments(
        run_name=args.experiment,
        report_to=None if args.debug else 'wandb',
        output_dir=weights_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,  # total number of optimization steps
        per_device_train_batch_size=8,  # batch size per device during training - change to 8 for bart-base
        per_device_eval_batch_size=8,  # batch size for evaluation - change to 8 for bart-base
        gradient_accumulation_steps=max(1, 8 // args.num_gpus),  # change to 8 for bart-base (effect bs of 64)
        eval_accumulation_steps=1,
        warmup_steps=0 if args.debug else 200,  # number of warmup steps for learning rate scheduler
        weight_decay=0,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        logging_first_step=True,
        no_cuda=args.cpu,
        fp16=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        learning_rate=5e-5,
        dataloader_num_workers=0 if args.debug else 8,
        label_smoothing_factor=0.1,
        save_strategy='steps',
        predict_with_generate=True,
        save_steps=999999 if args.debug else 5000,
        save_total_limit=3,
        seed=args.seed,
        evaluation_strategy='steps',
        eval_steps=999999 if args.debug else 1000,
        load_best_model_at_end=True,
        metric_for_best_model='mean_f1',
        greater_is_better=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest'
    )

    rouge = load_metric('rouge')

    def model_init():
        # kwargs = {'gradient_checkpointing': True}
        model = BartForConditionalGeneration.from_pretrained(args.hf_model)
        model.resize_token_embeddings(len(tokenizer))
        return model

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=rouge_types)

        stats = {}
        f1s = []
        for rouge_type in rouge_types:
            stats[f'{rouge_type}_precision'] = rouge_output[rouge_type].mid.precision
            stats[f'{rouge_type}_recall'] = rouge_output[rouge_type].mid.recall
            stats[f'{rouge_type}_f1'] = rouge_output[rouge_type].mid.fmeasure
            f1s.append(rouge_output[rouge_type].mid.fmeasure)

        stats['mean_f1'] = np.array(f1s).mean()
        return stats

    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=None if args.tune else model,  # the instantiated 🤗 Transformers model to be trained
        model_init=model_init if args.tune else None,
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.tune:
        def my_hp_space(trial):
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'warmup_steps': trial.suggest_categorical('warmup_steps', [25, 250, 500]),
                'max_steps': trial.suggest_categorical('max_steps', [10000, 25000, 50000]),
                # 'weight_decay': trial.suggest_float('weight_decay', 0.1, 0.3),
            }
        trainer.hyperparameter_search(
            direction='maximize',
            backend='optuna',
            n_trials=10,  # number of trials
            hp_space=my_hp_space
        )
    else:
        trainer.train()
