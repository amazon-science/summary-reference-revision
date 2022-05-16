# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import random
random.seed(1992)
import os
import ujson
import regex as re

import numpy as np
np.random.seed(1992)
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from comp_med_dsum_eval.preprocess.constants import HTML_REGEX_NO_SPACE
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import get_attr
from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments


ENT_ORDER = ['dx', 'procedure', 'treatment', 'test', 'med']
ENT_TYPE_MAP = {
    'DX_NAME': 'dx',
    'PROCEDURE_NAME': 'procedure',
    'TREATMENT_NAME': 'treatment',
    'TEST_NAME': 'test',
    'BRAND_NAME': 'med',
    'GENERIC_NAME': 'med'
}


def remove_tags_from_sent(str):
    return re.sub(HTML_REGEX_NO_SPACE, '', str)


def resolve_delta_bucket(delta):
    delta = max(0, delta)
    return str(int(min(delta // 5, 5)))


def tokenize(str):
    return list(filter(lambda x: len(x) > 0, re.split(r'\W+', remove_tags_from_sent(str))))


def resolve_bert_bucket(bs_coverage):
    cov = max(0, bs_coverage - 50)
    return str(int(cov // 5))


def resolve_coverage_bucket(extractive_coverage):
    return str(int(extractive_coverage // 0.1))


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def mask_covered_ents(html_str):
    tps = re.split(HTML_REGEX_NO_SPACE, html_str)
    output_str = ''
    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<e') or tp == '</e>':
            continue
        elif tp_idx > 0 and tps[tp_idx - 1].startswith('<e') and int(get_attr(tps[tp_idx - 1], 'halluc')) == 0:
            lead_space = re.search(r'^([\s]+)', tp)
            trail_space = re.search(r'([\s]+)$', tp)
            lead_space_str = '' if lead_space is None else lead_space.group(0)
            trail_space_str = '' if trail_space is None else trail_space.group(0)
            output_str += lead_space_str + '<mask>' + trail_space_str
        else:
            output_str += tp
    return output_str


class SingleBatchCollate:
    def __call__(self, batch):
       return batch[0]


class ReviseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, data_fn, tokenizer, debug=False, max_val_examples=1024,
                 denoise_only=False, contrast_only=False, pos_only=False, contrast_input_strategy='worst',
                 remove_redress=False, remove_same_sum=False, remove_codes=False):
        super().__init__()

        with open(data_fn, 'r') as fd:
            examples = ujson.load(fd)

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.debug = debug
        self.denoise_only = denoise_only
        self.contrast_only = contrast_only
        self.pos_only = pos_only
        self.remove_redress = remove_redress
        self.remove_same_sum = remove_same_sum
        self.remove_codes = remove_codes

        if self.debug:
            self.train_examples = self.val_examples = examples
        else:
            self.train_example_ids = set(pd.read_csv(os.path.join(data_dir, 'train_example_ids.csv'))['example_id'])
            self.val_example_ids = set(pd.read_csv(os.path.join(data_dir, 'validation_example_ids.csv'))['example_id'])
            self.train_examples = list(filter(lambda example: example['example_id'] in self.train_example_ids, examples))
            self.val_examples = list(filter(lambda example: example['example_id'] in self.val_example_ids, examples))
        if len(self.val_examples) > max_val_examples:
            self.val_examples = list(np.random.choice(self.val_examples, size=(max_val_examples,), replace=False))

        self.tokenizer = tokenizer
        self.num_workers = 0 if self.debug else 8
        self.max_val_examples = max_val_examples
        self.contrast_input_strategy = contrast_input_strategy

    def train_dataloader(self):
        kwargs = {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': self.num_workers,
            'collate_fn': SingleBatchCollate()
        }
        train_dataset = ReviseDataset(
            self.train_examples, self.tokenizer, split='train',
            denoise_only=self.denoise_only, contrast_only=self.contrast_only,
            contrast_input_strategy=self.contrast_input_strategy, pos_only=self.pos_only,
            remove_redress=self.remove_redress, remove_same_sum=self.remove_same_sum, remove_codes=self.remove_codes
        )
        return DataLoader(train_dataset, **kwargs)

    def predict_dataloader(self):
        pass

    def val_dataloader(self, max_num=512, is_eval=True):
        kwargs = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': self.num_workers,
            'collate_fn': SingleBatchCollate()
        }
        val_examples = self.val_examples[:min(len(self.val_examples), max_num)]
        val_dataset = ReviseDataset(
            val_examples, self.tokenizer, split='val', denoise_only=False,
            contrast_only=False, contrast_input_strategy='worst', pos_only=False,
            remove_redress=False, remove_same_sum=False, remove_codes=self.remove_codes
        )
        return DataLoader(val_dataset, **kwargs)


class ReviseDataset(Dataset):
    def __init__(self, examples, tokenizer, max_output_length=128, max_input_length=None, split='train',
                 denoise_only=False, contrast_only=False, contrast_input_strategy='worst', pos_only=False,
                 remove_redress=False, remove_same_sum=False, remove_codes=False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_output_length = max_output_length
        self.max_input_length = min(
            1024, self.tokenizer.model_max_length if max_input_length is None else max_input_length)
        self.pad_token_id = tokenizer.pad_token_id
        self.split = split
        self.denoise_only = denoise_only
        self.contrast_only = contrast_only
        self.pos_only = pos_only
        self.remove_redress = remove_redress
        self.remove_same_sum = remove_same_sum
        self.remove_codes = remove_codes
        self.contrast_input_strategy = contrast_input_strategy
        self.example_id_2_dataset_idx = defaultdict(list)
        for dataset_idx, example in enumerate(self.examples):
            self.example_id_2_dataset_idx[example['example_id']].append(dataset_idx)

    def __getitem__(self, idx):
        example = self.examples[idx]
        example_id = example['example_id']
        other_dataset_idxs = list(filter(
            lambda dataset_idx: dataset_idx != idx, self.example_id_2_dataset_idx[example_id]))
        if len(other_dataset_idxs) > 0:
            distractor_data_idx = int(np.random.choice(other_dataset_idxs, size=(1,), )[0])
        else:
            distractor_data_idx = int(np.random.randint(0, len(self), size=(1,))[0])
        source = example['source']
        source_sents = source['sents']
        target_sent = example['target_sent']
        target_sent_clean = remove_tags_from_sent(target_sent)

        source_toks = tokenize(' '.join(source_sents))
        target_toks = tokenize(target_sent_clean)
        source_extractive_frags = parse_extractive_fragments(source_toks, target_toks, remove_stop=True)

        # Distractor (noisy) input and context
        distractor_example = self.examples[distractor_data_idx]
        distractor_context = distractor_example['source']['sents']
        distractor_input = distractor_example['target_sent']
        distractor_input_toks = tokenize(distractor_input)
        distractor_context_clean = list(map(remove_tags_from_sent, distractor_context))
        distractor_context_toks = tokenize(' '.join(distractor_context))
        distractor_input_clean = remove_tags_from_sent(distractor_input)

        # Get alignment between distractor context and distractor inputs
        dd_frags = parse_extractive_fragments(distractor_context_toks, distractor_input_toks, remove_stop=True)
        dd_source_extract_bucket = resolve_coverage_bucket(dd_frags['coverage'])
        dd_source_extract_code = '' if self.remove_codes else f'<source-extract-{dd_source_extract_bucket}>'

        # Get alignment between this target (as input) and distractor sentence (as decoder output)
        cd_input_frags = parse_extractive_fragments(target_toks, distractor_input_toks, remove_stop=True)
        cd_input_extract_bucket = resolve_coverage_bucket(cd_input_frags['coverage'])
        cd_input_extract_code = '' if self.remove_codes else f'<input-extract-{cd_input_extract_bucket}>'

        # Get alignment between this target sentence (as decoder output) and distractor (as input)
        dc_input_frags = parse_extractive_fragments(distractor_input_toks, target_toks, remove_stop=True)
        dc_input_extract_bucket = resolve_coverage_bucket(dc_input_frags['coverage'])
        dc_input_extract_code = '' if self.remove_codes else f'<input-extract-{dc_input_extract_bucket}>'

        perturbs = example['perturb']
        perturb_sents, perturb_covs = perturbs['sents'], perturbs['source_to_perturb_coverage']

        # Generate Encoder Inputs
        # 1. masked target
        # 2. target
        # 3x. worst perturbation repeated
        masked_target = mask_covered_ents(target_sent)
        if self.contrast_input_strategy == 'worst':
            perturb_input_idx = int(np.argmin(perturb_covs))
        elif self.contrast_input_strategy == 'best':
            perturb_input_idx = int(np.argmax(perturb_covs))
        elif self.contrast_input_strategy == 'random':
            perturb_input_idx = int(np.random.randint(len(perturb_covs)))
        else:
            raise Exception(f'Unrecognized strategy --> {self.contrast_input_strategy}')
        perturb_target_idx = int(np.random.randint(len(perturb_covs)))
        perturb_target_sent = perturb_sents[perturb_target_idx]
        perturb_target_sent_clean = remove_tags_from_sent(perturb_target_sent)

        source_extract_bucket = resolve_coverage_bucket(source_extractive_frags['coverage'])
        source_extract_code = '' if self.remove_codes else f'<source-extract-{source_extract_bucket}>'

        perturb_toks = tokenize(perturb_sents[perturb_input_idx])
        perturb_input_clean = remove_tags_from_sent(perturb_sents[perturb_input_idx])
        perturb_frags = parse_extractive_fragments(perturb_toks, target_toks, remove_stop=True)
        perturb_input_extract_bucket = resolve_coverage_bucket(perturb_frags['coverage'])
        perturb_input_extract_code = '' if self.remove_codes else f'<input-extract-{perturb_input_extract_bucket}>'

        # Positive Inputs
        # distractor input, context -> target
        # perturb input, context -> target
        source_clean = [remove_tags_from_sent(x) for x in source_sents]
        context_input_str = ' '.join(source_clean)
        distractor_pos_input = dc_input_extract_code + source_extract_code + distractor_input_clean +\
            ' <sep> ' + context_input_str
        noise_pos_input = perturb_input_extract_code + source_extract_code + perturb_input_clean +\
            ' <sep> ' + context_input_str
        pos_inputs = [noise_pos_input, distractor_pos_input]

        # Negative Inputs
        distractor_context_input = cd_input_extract_code + dd_source_extract_code + target_sent_clean +\
            ' <sep> ' + ' '.join(distractor_context_clean)
        noise_neg_input = perturb_input_extract_code + source_extract_code + perturb_input_clean +\
            ' <sep> ' + context_input_str
        neg_inputs = [noise_neg_input, distractor_context_input]

        contrast_inputs = pos_inputs + neg_inputs
        # # ENCODER
        if self.denoise_only:
            inputs = [masked_target, perturb_input_clean]
        else:
            inputs = [masked_target] + contrast_inputs
        model_inputs = self.tokenizer(
            inputs,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        # DECODER
        if self.denoise_only:
            targets = [target_sent_clean] * 2
        else:
            targets = [target_sent_clean] * 4 + [perturb_target_sent_clean]

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=self.max_output_length,
                return_tensors='pt',
            )['input_ids']
            labels[torch.where(labels == self.pad_token_id)] = -100
            model_inputs['labels'] = labels

        if self.denoise_only:
            model_inputs['mask_idxs'] = torch.LongTensor([0, 1])
        else:
            # Assign positive and negative samples
            if self.remove_same_sum:
                pos_idxs, neg_idxs = [1], [3]
            elif self.remove_redress:
                pos_idxs, neg_idxs = [2], [4]
            else:
                pos_idxs, neg_idxs = [1, 2], [3, 4]
            if not self.contrast_only:  # contrast_only is a baseline which ablates the entity unmasking objective
                model_inputs['mask_idxs'] = torch.LongTensor([0])
            model_inputs['pos_idxs'] = torch.LongTensor(pos_idxs)
            if not self.pos_only:
                model_inputs['neg_idxs'] = torch.LongTensor(neg_idxs)
        return model_inputs

    def __len__(self):
        return len(self.examples)


class GenerateDataset(Dataset):
    def __init__(self, examples, tokenizer, max_output_length=128, max_input_length=None, remove_codes=False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_output_length = max_output_length
        self.max_input_length = min(
            1024, self.tokenizer.model_max_length if max_input_length is None else max_input_length)
        self.pad_token_id = tokenizer.pad_token_id
        self.remove_codes = remove_codes

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target_sent = example['target_sent']
        source_sents = source['sents']
        source_sent_idxs = source['sent_idxs']

        # target_cov = example['source_to_target_coverage']
        # target_bucket = int(resolve_bert_bucket(target_cov * 100))
        source_toks = list(map(lambda x: x.lower(), tokenize(' '.join(source_sents))))
        target_sent_clean = remove_tags_from_sent(target_sent)
        target_toks = list(map(lambda x: x.lower(), tokenize(target_sent_clean)))

        # The "target" (summary) sentence, here, is the reference sentence
        input_extractive_frags = parse_extractive_fragments(source_toks, target_toks, remove_stop=True)
        input_extract_bucket = resolve_coverage_bucket(input_extractive_frags['coverage'])

        if self.remove_codes:
            prefix_codes = ['']  # No prefix codes (1 generation)
            meta_codes = [{'input_extract_code': -1, 'source_extract_code': -1}]  # Dummy Values
        else:
            prefix_codes = []
            meta_codes = []
            for j in range(11):
                meta_codes.append({
                    'input_extract_code': int(input_extract_bucket),
                    'source_extract_code': j
                })
                prefix_codes.append(f'<input-extract-{input_extract_bucket}><source-extract-{j}>')

        # ENCODER
        # Generate Encoder Inputs
        target_input = remove_tags_from_sent(target_sent)
        context_str = ' '.join([remove_tags_from_sent(x) for x in source_sents])

        model_input_strs = [prefix_code + target_input + ' <sep> ' + context_str for prefix_code in prefix_codes]
        model_inputs = self.tokenizer(
            model_input_strs,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        meta = {}
        for k, v in example.items():
            if type(v) != dict:
                meta[k] = v
        str_inputs = {
            'context': ''.join([
                f'<s idx={sent_idx}>{body}</s>' for sent_idx, body in zip(source_sent_idxs, source_sents)]),
            'meta_codes': meta_codes
        }
        return model_inputs, str_inputs, meta

    def __len__(self):
        return len(self.examples)
