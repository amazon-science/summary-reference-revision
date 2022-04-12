# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ujson
import regex as re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent, get_attr, remove_empty
from comp_med_dsum_eval.preprocess.entity.process_ents import filter_ents, add_ent_id


def mark_halluc_ents(ent_obj, matched_ent_ids):
    annotated_sents = []
    sent_ents = [x for x in ent_obj if x['dtype'] == 'sent']
    for sent_obj in sent_ents:
        annotated = ''
        prev_start = 0
        sent_text = sent_obj['text']
        for ent in filter_ents(add_ent_id(sent_obj, 'target', return_ents=True)):
            start, end = ent['BeginOffset'], ent['EndOffset']
            prev_chunk = sent_obj['text'][prev_start:start]
            prev_end_in_space = re.search(r'([\s]+)$', prev_chunk)
            annotated += prev_chunk.rstrip()
            ent_id = ent['ent_id']
            ent_space_pre = '' if prev_end_in_space is None else prev_end_in_space.group(0)
            halluc = ent_id not in matched_ent_ids
            if halluc:
                annotated += '<halluc>' + ent_space_pre + sent_text[start:end] + '</halluc>'
            else:
                annotated += ent_space_pre + sent_text[start:end]
            prev_start = end
        annotated += sent_text[prev_start:] + ' '
        annotated_sents.append(annotated)
    return ' '.join(annotated_sents)


def add_meta(data_df, data_dir):
    data_df = data_df.assign(
        patient_id=data_df['example_id'].apply(lambda x: x.split('_')[0]),
        visit_id=data_df['example_id'].apply(lambda x: x.split('_')[1]),
    )
    train_example_ids = set(pd.read_csv(os.path.join(data_dir, 'train_example_ids.csv'))['example_id'])
    val_example_ids = set(pd.read_csv(os.path.join(data_dir, 'validation_example_ids.csv'))['example_id'])
    test_example_ids = set(pd.read_csv(os.path.join(data_dir, 'test_example_ids.csv'))['example_id'])
    low_cov_fn = os.path.join(data_dir, 'low_coverage_examples.csv')
    if os.path.exists(low_cov_fn):
        low_quality_examples = set(pd.read_csv(os.path.join(data_dir, 'low_coverage_examples.csv'))['example_id'])
        mid_quality_examples = set(pd.read_csv(os.path.join(data_dir, 'mid_coverage_examples.csv'))['example_id'])
        high_quality_examples = set(pd.read_csv(os.path.join(data_dir, 'high_coverage_examples.csv'))['example_id'])
    else:
        low_quality_examples = mid_quality_examples = high_quality_examples = {}

    # TODO make this path adjustable -- not hardcoded to /efs/griadams
    viable_visits = set(pd.read_csv('/efs/griadams/viable_visits.csv')['HADM_ID'].apply(
        lambda x: str(x).split('.')[0]))

    def get_split(example_id):
        if example_id in train_example_ids:
            return 'train'
        elif example_id in val_example_ids:
            return 'validation'
        elif example_id in test_example_ids:
            return 'test'
        else:
            raise Exception(f'Unassigned example id -> {example_id}')

    def get_quality(example_id):
        if example_id in mid_quality_examples:
            return 'mid'
        elif example_id in low_quality_examples:
            return 'low'
        elif example_id in high_quality_examples:
            return 'high'
        else:
            return None  # Exception(f'Unassigned example id -> {example_id}')
    data_df = data_df.assign(
        split=data_df['example_id'].apply(get_split),
        quality=data_df['example_id'].apply(get_quality),
        has_admission_note=data_df['visit_id'].apply(lambda visit_id: visit_id in viable_visits)
    )
    return data_df


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer, for_eval_only=False, max_val_num=None, ignore_hallucinated_ents=False):
        super().__init__()

        mini_str = '_mini' if args.debug else ''
        data_suffix = '' if args.version == 'original' else f'_{args.reviser_experiment}_{args.version}'
        data_fn = os.path.join(args.data_dir, f'summary_dataset_rouge_annotated{data_suffix}{mini_str}.csv')
        print(f'Reading in dataset from {data_fn}')
        data_df = pd.read_csv(data_fn)
        self.max_val_num = max_val_num
        if not for_eval_only:
            self.require_admission_note = args.require_admission_note
            self.remove_low = args.low_weight == 0
            self.remove_mid = args.mid_weight == 0
            self.remove_high = args.high_weight == 0
            assert not (self.remove_low and self.remove_mid and self.remove_high)
        self.data_df = add_meta(data_df, data_dir=args.data_dir)

        num_val = len(self.data_df[self.data_df['split'] == 'validation'])
        if num_val == 0:
            data_fn = os.path.join(args.data_dir, f'summary_dataset_rouge_annotated{mini_str}.csv')
            original_data_df = pd.read_csv(data_fn)
            original_data_df = add_meta(original_data_df, data_dir=args.data_dir)
            non_train_df = original_data_df[original_data_df['split'].isin({'validation', 'test'})]
            print(f'Concatenating {len(non_train_df)} original non-training examples to revised training set of'
                  f' size {len(self.data_df)}')
            self.data_df = pd.concat([self.data_df, non_train_df])

        self.ignore_hallucinated_ents = ignore_hallucinated_ents
        self.control_hallucinations = args.control_hallucinations

        self.data_dir = args.data_dir
        self.debug = args.debug
        self.tokenizer = tokenizer
        self.hf_model = args.hf_model
        self.num_workers = 0 if self.debug else 8
        self.max_input_length = tokenizer.model_max_length if args.max_input_length is None else args.max_input_length
        if self.max_input_length > tokenizer.model_max_length:
            print(f'Warning! Setting maximum input length to be maximum model length of {tokenizer.model_max_length}')
            self.max_input_length = tokenizer.model_max_length
        self.max_output_length = 1024

    def train_dataloader(self):
        train_df = self.data_df[self.data_df['split'] == 'train']
        # train_df.assign(valid=train_df['target'].apply(lambda x: '</s>' in x))  # > 0 sentences in reference
        # pre_n = len(train_df)
        # train_df = train_df[~train_df['valid']]
        # n = len(train_df)
        # print(f'{pre_n - n}/{pre_n} training examples have empty references...')
        if self.remove_low:
            print('Filtering out low quality examples from train dataset...')
            train_df = train_df[train_df['quality'] != 'low']
        if self.remove_mid:
            print('Filtering out mid quality examples from train dataset...')
            train_df = train_df[train_df['quality'] != 'mid']
        if self.remove_high:
            print('Filtering out high quality examples from train dataset...')
            train_df = train_df[train_df['quality'] != 'high']
        if self.require_admission_note:
            print('Filtering for examples with an admission note...')
            train_df = train_df[train_df['has_admission_note']]
        records = train_df.to_dict('records')
        train_split = SummarizationDataset(
            records, 'train', self.max_input_length, ignore_hallucinated_ents=self.ignore_hallucinated_ents,
            control_hallucinations=self.control_hallucinations, data_dir=self.data_dir
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            add_global_att='led' in self.hf_model,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            ignore_hallucinated_ents=self.ignore_hallucinated_ents,
        )
        kwargs = {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(train_split, **kwargs)

    def test_dataloader(self, add_cols=None):
        test_df = self.data_df[self.data_df['split'] == 'test']
        records = test_df.to_dict('records')
        test_split = SummarizationDataset(
            records, 'test', self.max_input_length, control_hallucinations=self.control_hallucinations, is_eval=True
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            add_global_att='led' in self.hf_model,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_cols=add_cols
        )
        kwargs = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(test_split, **kwargs)

    def val_dataloader(self, max_n=None, add_cols=None):
        val_df = self.data_df[self.data_df['split'] == 'validation']
        if self.remove_low:
            print('Filtering out low quality examples from validation dataset...')
            val_df = val_df[val_df['quality'] != 'low']
        if self.remove_mid:
            print('Filtering out mid quality examples from validation dataset...')
            val_df = val_df[val_df['quality'] != 'mid']
        if self.remove_high:
            print('Filtering out high quality examples from validation dataset...')
            val_df = val_df[val_df['quality'] != 'high']
        if self.require_admission_note:
            print('Filtering for examples with an admission note...')
            val_df = val_df[val_df['has_admission_note']]
        max_n = min(filter(None, [max_n, self.max_val_num]))
        if max_n is not None and max_n < len(val_df):
            print(f'Sampling {max_n} examples out of {len(val_df)}')
            val_df = val_df.sample(n=max_n, replace=False, random_state=1992)
        records = val_df.to_dict('records')
        val_split = SummarizationDataset(
            records, 'validation', self.max_input_length, control_hallucinations=self.control_hallucinations,
            is_eval=True
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            add_global_att='led' in self.hf_model,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_cols=add_cols
        )
        kwargs = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(val_split, **kwargs)


class Seq2SeqCollate:
    def __init__(
            self, tokenizer, add_global_att, max_input_length=8192, max_output_length=512, add_cols=None,
            ignore_hallucinated_ents=False
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.max_output_length = max_output_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.add_global_att = add_global_att
        self.add_cols = [] if add_cols is None else add_cols
        self.ignore_hallucinated_ents = ignore_hallucinated_ents

    def tokenize_with_ignore_labels(self, texts, max_length=None):
        ids, ignore_idxs = [], []
        bsize = len(texts)
        for batch_idx, text in enumerate(texts):
            id_set = [self.tokenizer.bos_token_id]
            halluc_splits = re.split(r'(</?halluc>)', text)
            for tp_idx, tp in enumerate(halluc_splits):
                if tp in {'<halluc>', '</halluc>'}:
                    continue
                id_arr = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tp))
                if tp_idx > 0 and halluc_splits[tp_idx - 1] == '<halluc>':
                    for ignore_idx in range(len(id_set), len(id_set) + len(id_arr)):
                        if ignore_idx < max_length:
                            ignore_idxs.append((batch_idx, ignore_idx))
                id_set += id_arr

            trunc_n = min(len(id_set), max_length - 1)
            ids.append(id_set[:trunc_n] + [self.tokenizer.eos_token_id])
        lens = [len(id) for id in ids]
        max_len = max(lens)
        padded_ids = np.zeros([bsize, max_len], dtype=np.int64)
        padded_ids.fill(self.tokenizer.pad_token_id)
        for batch_idx, id in enumerate(ids):
            padded_ids[batch_idx, :len(id)] = id
        return torch.from_numpy(padded_ids), ignore_idxs

    def __call__(self, batch_list):
        # tokenize the inputs and labels
        inputs = self.tokenizer(
            [x['source'] for x in batch_list],
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        batch = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
        }

        if self.ignore_hallucinated_ents:
            labels, ignore_idxs = self.tokenize_with_ignore_labels(
                [x['target'] for x in batch_list],
                max_length=self.max_output_length,
            )
            batch['ignore_idxs'] = ignore_idxs
        else:
            labels = self.tokenizer(
                [x['target'] for x in batch_list],
                padding='longest',
                truncation=True,
                max_length=self.max_output_length,
                return_tensors='pt'
            ).input_ids

        if self.add_global_att:
            # create 0 global_attention_mask lists
            batch['global_attention_mask'] = torch.FloatTensor(len(batch['input_ids']) * [
                [0 for _ in range(len(batch['input_ids'][0]))]
            ])

            # the 1st element of each sequence in batch should be flipped to 1
            batch['global_attention_mask'][:, 0] = 1

        batch['labels'] = labels
        # We have to make sure that the PAD token is ignored
        batch['labels'][torch.where(batch['labels'] == 1)] = -100
        batch['quality'] = [x['quality'] for x in batch_list]
        batch['has_admission_note'] = [x['has_admission_note'] for x in batch_list]
        for col in self.add_cols:
            batch[col] = [x[col] for x in batch_list]
        return batch


def truncate_source(source_html, max_len, wp_multiplier=1.3):
    target_toks = round(max_len / wp_multiplier)
    tps = source_html.split('<SEP>')
    sent_info = []

    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<s'):
            sent_idx = get_attr(tp, 'idx')
            num_toks = len(remove_tags_from_sent(tps[tp_idx + 1]).split(' '))
            rank = int(get_attr(tp, 'rank'))
            sent_info.append({
                'rank': rank,
                'sent_idx': sent_idx,
                'num_toks': num_toks
            })

    keep_sent_idxs = []
    curr_len = 0
    sent_info = list(sorted(sent_info, key=lambda x: x['rank']))
    for sent in sent_info:
        curr_len += sent['num_toks']
        keep_sent_idxs.append(int(sent['sent_idx']))
        if curr_len >= target_toks:
            break
    keep_sent_idxs = set(keep_sent_idxs)
    keep_tps = []
    for tp_idx, tp in enumerate(tps):
        if tp_idx > 0 and tps[tp_idx - 1].startswith('<s') and \
                int(get_attr(tps[tp_idx - 1], 'idx')) not in keep_sent_idxs:
            continue
        keep_tps.append(tp)

    keep_tps = remove_empty(keep_tps, 's')
    keep_tps = remove_empty(keep_tps, 'p')
    keep_tps = remove_empty(keep_tps, 'h')
    return '<SEP>'.join(keep_tps)


class SummarizationDataset(Dataset):
    def __init__(
            self, examples, split, max_input_length, ignore_hallucinated_ents=False, control_hallucinations=False,
            data_dir=None, is_eval=False
    ):
        super(SummarizationDataset, self).__init__()
        self.examples = examples
        self.split = split
        self.max_input_length = max_input_length
        self.ignore_hallucinated_ents = ignore_hallucinated_ents
        self.data_dir = data_dir
        self.control_hallucinations = control_hallucinations
        self.is_eval = is_eval

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        source_text = example['source']
        truncated_source = remove_tags_from_sent(
            truncate_source(source_text, self.max_input_length), include_headers=True, include_subheaders=True)
        if self.control_hallucinations:
            quality = example['quality']
            # TODO. Eventually fix the names.  high-halluc actually means high quality
            prefix = '<high-halluc>' if self.is_eval else f'<{quality}-halluc>'
            truncated_source = prefix + truncated_source
        example['source'] = truncated_source

        if self.ignore_hallucinated_ents:
            example_id = example['example_id']
            ent_fn = os.path.join(self.data_dir, 'acm_output', f'{example_id}.json')
            ent_merge_fn = os.path.join(self.data_dir, 'acm_output', f'{example_id}.csv')

            with open(ent_fn) as fd:
                ents = ujson.load(fd)
            target_ents = ents['target']
            try:
                reg_ent_merges = pd.read_csv(ent_merge_fn)
                reg_ent_merges = reg_ent_merges[reg_ent_merges['should_merge']]
                reg_ent_merges = reg_ent_merges[reg_ent_merges['source_ent_id'].apply(lambda x: 'sent' in x)]
                if len(reg_ent_merges) == 0:
                    matched_ent_ids = {}
                else:
                    matched_ent_ids = set(reg_ent_merges['target_ent_id'].unique())
            except:
                matched_ent_ids = {}
            if len(target_ents) > 0:
                mod_target = mark_halluc_ents(target_ents, matched_ent_ids)
            else:
                mod_target = remove_tags_from_sent(example['target'])
        else:
            mod_target = remove_tags_from_sent(example['target'])
        example['target'] = mod_target
        return example
