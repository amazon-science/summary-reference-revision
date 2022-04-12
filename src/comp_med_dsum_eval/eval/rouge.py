# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from string import punctuation

from lexrank import STOPWORDS
import numpy as np
from nltk import word_tokenize
from rouge_score import rouge_scorer


stopwords = STOPWORDS['en']
stopwords = stopwords.union(set([x for x in punctuation]))
DEFAULT_TYPES = ['rouge1', 'rouge2', 'rougeL']


def aggregate_f1(scores):
    return np.mean([x.fmeasure for x in scores.values()])


def calc_rouge(summary, reference, rouge_types=DEFAULT_TYPES):
    scorer = rouge_scorer.RougeScorer(rouge_types)
    scores = scorer.score(summary, reference)
    return scores


def preprocess_sentence(text, vocab_filter=None):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token.strip() for token in word_tokenize(text)]
    tokens_filt = [t for t in tokens if t not in punctuation and t not in stopwords and len(t.strip()) > 0]
    if vocab_filter is not None:
        tokens_filt = [t for t in tokens_filt if t.strip() in vocab_filter]
    return ' '.join(tokens_filt)


def top_rouge_sents(target, source_sents, rouge_types):
    n = len(source_sents)
    target_no_stop = preprocess_sentence(target)
    source_sents_no_stop = list(map(preprocess_sentence, source_sents))
    references = [target_no_stop for _ in range(n)]
    outputs = [calc_rouge(a, b) for a, b in zip(source_sents_no_stop, references)]
    scores = np.array([sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])

    sent_order = scores.argsort()[::-1]
    rouges = [scores[i] for i in sent_order]

    return sent_order, rouges


def max_rouge_set(target, source_sents, rouge_types, target_tok_ct=None):
    n = len(source_sents)
    target_no_stop = preprocess_sentence(target)
    source_sents_no_stop = list(map(preprocess_sentence, source_sents))
    curr_sum = ''
    curr_rouge = 0.0
    sent_order = []
    rouges = []
    metric = 'f1' if target_tok_ct is None else 'recall'
    for _ in range(n):
        _, idx, score = max_rouge_sent(
            target_no_stop, source_sents_no_stop, rouge_types, return_score=True, source_prefix=curr_sum,
            mask_idxs=sent_order, metric=metric
        )

        decreasing_score = score <= curr_rouge
        mc = target_tok_ct is not None and len(source_sents[idx].split(' ')) + len(curr_sum.split(' ')) > target_tok_ct

        if decreasing_score or mc:
            break
        curr_rouge = score
        curr_sum += source_sents[idx] + ' '
        sent_order.append(idx)
        rouges.append(curr_rouge)
    return sent_order, rouges


def max_rouge_sent(target, source_sents, rouge_types, return_score=False, source_prefix='', mask_idxs=[], metric='f1'):
    n = len(source_sents)
    predictions = [source_prefix + s for s in source_sents]
    references = [target for _ in range(n)]
    outputs = [calc_rouge(a, b) for a, b in zip(predictions, references)]
    if metric == 'f1':
        scores = np.array(
            [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'recall':
        scores = np.array(
            [sum([outputs[t][i].recall for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'precision':
        scores = np.array(
            [sum([outputs[t][i].precision for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    if len(mask_idxs) > 0:
        scores[mask_idxs] = float('-inf')
    max_idx = np.argmax(scores)
    max_source_sent = source_sents[max_idx]
    if return_score:
        return max_source_sent, max_idx, scores[max_idx]
    return max_source_sent, max_idx
