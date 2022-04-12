# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

import argparse
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent, sents_from_html


def nsp(s1, s2):
    encoding = tokenizer(s1, s2, return_tensors='pt', max_length=512, truncation=True)
    encoding = {k: v.cuda() for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding, labels=dummy_label)
    logits = outputs.logits
    prob = torch.softmax(logits, dim=1)
    return float(prob[0, 0])


def process(example_id, target):
    target_sents = [remove_tags_from_sent(x) for x in sents_from_html(target)]
    n = len(target_sents)
    if n <= 1:
        return {'example_id': example_id, 'nsp': None}
    nsps = []
    for i in range(n - 1):
        nsps.append(nsp(target_sents[i], target_sents[i + 1]))
    return {'example_id': example_id, 'nsp': sum(nsps) / float(len(nsps))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate NSP scores')
    parser.add_argument(
        '--pretrained_model', default='emilyalsentzer/Bio_ClinicalBERT', choices=['emilyalsentzer/Bio_ClinicalBERT'])
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--max_n', default=-1, type=int)

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    print('Loading model...')
    model = BertForNextSentencePrediction.from_pretrained(args.pretrained_model, return_dict=True).eval().cuda()
    dummy_label = torch.LongTensor([1]).cuda()

    in_fn = os.path.join(data_dir, 'summary_dataset_ent.csv')
    df = pd.read_csv(in_fn)
    inputs = df[['example_id', 'target']].to_dict('records')
    n = len(inputs)
    print('Generating NSP predictions for {} examples'.format(n))
    corel_df = pd.DataFrame(list(tqdm(map(lambda x: process(**x), inputs), total=n)))
    out_fn = os.path.join(data_dir, 'nsp_scores.csv')
    corel_df.to_csv(out_fn, index=False)
