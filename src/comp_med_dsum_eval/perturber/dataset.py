# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import regex as re
from random import gauss, shuffle

import numpy as np
from torch.utils.data import Dataset

from comp_med_dsum_eval.preprocess.entity.entity_utils import extract_ents


SPAN_SIZE_REMOVE_FRAC = 0.33
ENT_REMOVE_FRAC = 0.5
MEAN_SHUFFLE_ORDERLINESS = 0.5
ENT_ADD_FRAC = 0.5
ENT_ORDER = ['dx', 'procedure', 'treatment', 'test', 'med']
ENT_TYPE_MAP = {
    'DX_NAME': 'dx',
    'PROCEDURE_NAME': 'procedure',
    'TREATMENT_NAME': 'treatment',
    'TEST_NAME': 'test',
    'BRAND_NAME': 'med',
    'GENERIC_NAME': 'med'
}


def random_span(n, size):
    assert 0 <= size <= n
    candidate_spans = [(i, i + size) for i in range(0, n - size + 1)]
    return candidate_spans[np.random.choice(range(len(candidate_spans)), size=(1,))[0]]


def insert_ents_randomly(toks, ents_to_add):
    for ent in ents_to_add:
        idx = int(np.random.randint(0, len(toks) + 1))
        toks = toks[:idx] + ent.split(' ') + toks[idx:]
    return toks


def filter_no_ents(examples):
    return list(filter(lambda example: '</e>' in example['text'], examples))


def ent_prefix(type2ents):
    outputs = []
    for ent_type in ENT_ORDER:
        if ent_type not in type2ents:
            continue
        ent_str = ' <ent-sep> '.join(list(set(type2ents[ent_type])))
        outputs.append(
            f'<{ent_type}> {ent_str} </{ent_type}>'
        )
    return ' '.join(outputs)


def compute_prefix(ents, ent2type, ent_mask_n, ent_add_n, span_mask_n, orderliness, max_val=10, add_control=True):
    ent_mask_n = min(max_val, ent_mask_n)
    ent_add_n = min(max_val, ent_add_n)
    orderliness = int(round(orderliness, 1) * 10)

    type2ents = defaultdict(list)
    for ent in ents:
        type2ents[ent2type[ent]].append(ent)
    ent_str = ent_prefix(type2ents)
    control_str = ''
    if add_control:
        control_str += f'<ent-remove-{ent_mask_n}> <ent-add-{ent_add_n}> <span-remove-{span_mask_n}> ' \
                   f'<shuffle-{orderliness}> <sep> '
    return control_str + ent_str + ' <sep> ', control_str


def compute_simple_prefix(span_mask_n, orderliness, max_val=10):
    orderliness = int(round(orderliness, 1) * max_val)
    return f'<span-remove-{span_mask_n}> <shuffle-{orderliness}> <sep> '


def partial_shuffle(arr, orderliness=0.5):
    # https://stackoverflow.com/questions/62436299/how-to-lightly-shuffle-a-list-in-python
    def _tuplify(x, y):
        return orderliness * y + gauss(0, 1), x

    pairs = list(map(_tuplify, arr, range(len(arr))))
    pairs.sort()
    partially_ordered_values = [p[1] for p in pairs]
    return partially_ordered_values


def random_span_removal(arr, size):
    if size == 0:
        return arr
    span_to_remove = random_span(len(arr), size)
    first_half = arr[:span_to_remove[0]]
    second_half = arr[span_to_remove[1]:] if span_to_remove[1] < len(arr) else []
    full = first_half + second_half
    assert len(arr) - len(full) == size
    return full


def random_span_addition(arr, size):
    span_to_add = random_span(len(arr), size)
    return arr[span_to_add[0]: span_to_add[1]]


def generate_perturb_input_simple(text):
    target, perturbed, _, _, _ = extract_ents(text, set())
    sampled_frac = np.random.beta(1, 1 / SPAN_SIZE_REMOVE_FRAC - 1)
    perturbed_toks = [x for x in perturbed.split(' ') if len(x) > 0]
    span_to_remove_len = round(sampled_frac * len(perturbed_toks))
    perturbed_toks = random_span_removal(perturbed_toks, span_to_remove_len)
    chosen_orderliness = np.random.beta(1, 1)
    perturbed_toks = partial_shuffle(perturbed_toks, orderliness=chosen_orderliness)
    prefix = compute_simple_prefix(span_to_remove_len, chosen_orderliness)
    source = prefix + ' '.join(perturbed_toks)
    return source, target


def extract_related_ents(example, max_ent_type=None):
    flat = []
    ent2type = {}
    for ent_type in ENT_ORDER:
        type_ents = example[ent_type]
        if type(type_ents) == str and len(type_ents) > 0:
            ent_arr = type_ents.split('<SEP>')
            if max_ent_type is not None and len(ent_arr) > max_ent_type:
                ent_arr = list(np.random.choice(ent_arr, size=(max_ent_type, ), replace=False))
            flat += ent_arr
            for ent in ent_arr:
                ent2type[ent] = ent_type
    return flat, ent2type


def generate_perturb_input(text, example, include_pos_ent=True, add_control=True, max_ent_type=None):
    num_ents = example['num_ents']
    num_ents_to_remove = round(np.random.beta(1, 1 / ENT_REMOVE_FRAC - 1) * num_ents)
    ent_idxs_to_remove = set(list(np.random.choice(np.arange(num_ents), size=(num_ents_to_remove,), replace=False)))

    target, perturbed, removed_ents, removed_ent_types, text_ents = extract_ents(text, ent_idxs_to_remove)
    text_ents_lower = set([x.lower() for x in text_ents])

    related_ents, ent2type = extract_related_ents(example, max_ent_type=max_ent_type)
    related_ents = list(set([e for e in related_ents if e.lower() not in text_ents_lower]))
    num_related = len(related_ents)

    num_ents_to_add = min(num_related, round(np.random.beta(1, 1 / ENT_ADD_FRAC - 1) * num_ents))
    if num_ents_to_add == 0:
        ent_idxs_to_add = set()
    else:
        ent_idxs_to_add = set(list(np.random.choice(
            np.arange(num_related), size=(num_ents_to_add,), replace=False)))
    ents_to_add = [ent for i, ent in enumerate(related_ents) if i in ent_idxs_to_add]
    prefix_ents = [ent for i, ent in enumerate(related_ents) if i not in ent_idxs_to_add]
    if include_pos_ent:  # should we include the entities we removed as part of the prefix text or not?
        # this should be True for training and False during inference
        prefix_ents += removed_ents
        for removed_ent, ent_type in zip(removed_ents, removed_ent_types):
            ent2type[removed_ent] = ent_type
    shuffle(prefix_ents)
    perturbed_toks = [x for x in perturbed.split(' ') if len(x) > 0]
    num_toks_remaining = len(perturbed_toks)

    sampled_frac = np.random.beta(1, 1 / SPAN_SIZE_REMOVE_FRAC - 1)
    span_to_remove_len = round(sampled_frac * num_toks_remaining)
    perturbed_toks = random_span_removal(perturbed_toks, span_to_remove_len)
    perturbed_toks = insert_ents_randomly(perturbed_toks, ents_to_add)
    chosen_orderliness = np.random.beta(1, 1)
    perturbed_toks = partial_shuffle(perturbed_toks, orderliness=chosen_orderliness)

    prefix, control_str = compute_prefix(
        prefix_ents, ent2type, num_ents_to_remove, num_ents_to_add, span_to_remove_len, chosen_orderliness,
        max_val=10, add_control=add_control
    )
    source = prefix + ' '.join(perturbed_toks)
    return source, target


class PerturbDataset(Dataset):
    def __init__(
            self, split_df, tokenizer, max_output_length=128, max_input_length=None, no_ent=False,
            max_ent_type=10
    ):
        self.examples = filter_no_ents(split_df.to_dict('records'))
        self.tokenizer = tokenizer
        self.max_output_length = max_output_length
        self.max_input_length = min(
            1024, self.tokenizer.model_max_length if max_input_length is None else max_input_length)
        self.no_ent = no_ent
        self.max_ent_type = max_ent_type

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        if self.no_ent:
            source, target = generate_perturb_input_simple(text)
        else:
            source, target = generate_perturb_input(
                text, example, include_pos_ent=True, add_control=True, max_ent_type=self.max_ent_type)

        model_inputs = self.tokenizer(
            source,
            add_special_tokens=True,
            padding='do_not_pad',
            truncation=True,
            max_length=self.max_input_length,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                add_special_tokens=True,
                padding='do_not_pad',
                truncation=True,
                max_length=self.max_output_length,
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    def __len__(self):
        return len(self.examples)


class AlterCodeGenerateDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length=None, samples=5, max_ent_type=10):
        self.examples = filter_no_ents(df.to_dict('records'))
        self.tokenizer = tokenizer
        self.max_input_length = min(
            1024, self.tokenizer.model_max_length if max_input_length is None else max_input_length)
        print(f'{len(self)} sentences remaining after filtering out those with no extracted entities.')
        self.samples = samples
        self.max_ent_type = max_ent_type

    def __getitem__(self, idx):
        example = self.examples[idx]
        if 'num_ents' not in example:
            example['num_ents'] = len(re.findall(r'</e>', example['text']))
            assert example['num_ents'] > 0
        # TODO backward compatibility.  remove when re-running
        if 'sent_idx' not in example:
            example['sent_idx'] = example['sent_uid'].split('.')[1]

        text = example['text']
        target, perturbed, removed_ents, removed_ent_types, text_ents = extract_ents(text, set())
        text_ents_lower = set([x.lower() for x in text_ents])

        related_ents, ent2type = extract_related_ents(example, max_ent_type=self.max_ent_type)
        related_ents = list(set([e for e in related_ents if e.lower() not in text_ents_lower]))

        shuffle(related_ents)
        perturbed_toks = [x for x in perturbed.split(' ') if len(x) > 0]
        num_toks_remaining = len(perturbed_toks)

        sampled_frac = np.random.beta(1, 1 / SPAN_SIZE_REMOVE_FRAC - 1)
        span_to_remove_len = round(sampled_frac * num_toks_remaining)
        perturbed_toks = random_span_removal(perturbed_toks, span_to_remove_len)
        chosen_orderliness = np.random.beta(1, 1)
        # perturbed_toks = partial_shuffle(perturbed_toks, orderliness=chosen_orderliness)

        sources = []
        for i in range(1, self.samples, 1):
            prefix, control_str = compute_prefix(
                related_ents, ent2type, i, i, span_to_remove_len, chosen_orderliness,
                max_val=10, add_control=True
            )
            source = prefix + ' '.join(perturbed_toks)
            sources.append(source)

        model_inputs = self.tokenizer(
            sources,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        meta = {'example_id': example['example_id'], 'sent_idx': example['sent_idx'], 'text_original': text}
        return model_inputs, meta

    def __len__(self):
        return len(self.examples)


class SampleGenerateDataset(Dataset):
    def __init__(self, df, tokenizer, samples=5, no_ent=False, max_ent_type=10, ent_ctrl_add=1, no_ent_trick=False):
        self.examples = filter_no_ents(df.to_dict('records'))
        self.tokenizer = tokenizer
        self.samples = samples
        self.no_ent = no_ent
        self.max_ent_type = max_ent_type
        self.include_pos_ent = no_ent_trick  # do we include removed entities ('pos') back into the distractor
        self.ent_ctrl_add = ent_ctrl_add

    def __getitem__(self, idx):
        example = self.examples[idx]
        if 'num_ents' not in example:
            example['num_ents'] = len(re.findall(r'</e>', example['text']))
            assert example['num_ents'] > 0
        # TODO backward compatibility.  remove when re-running
        if 'sent_idx' not in example:
            example['sent_idx'] = example['sent_uid'].split('.')[1]

        text = example['text']
        if self.no_ent:
            sources = [generate_perturb_input_simple(text)[0] for _ in range(self.samples)]
        else:
            sources = [generate_perturb_input(
                text, example, include_pos_ent=self.include_pos_ent, add_control=True, max_ent_type=self.max_ent_type
            )[0] for _ in range(self.samples)]

            if self.ent_ctrl_add > 0:
                # Add one to entity control codes to encourage more perturbing
                sources = [
                    re.sub(
                        r'<ent-remove-(\d)>',
                        lambda x: '<ent-remove-' + str(int(x.group(1)) + self.ent_ctrl_add) + '>', x
                    ) for x in sources
                ]
                sources = [
                    re.sub(
                        r'<ent-add-(\d)>', lambda x: '<ent-add-' + str(int(x.group(1)) + self.ent_ctrl_add) + '>', x
                    ) for x in sources
                ]

        return {'inputs': sources, 'example_id': example['example_id'], 'sent_idx': example['sent_idx'],
                'text_original': text}

    def __len__(self):
        return len(self.examples)
