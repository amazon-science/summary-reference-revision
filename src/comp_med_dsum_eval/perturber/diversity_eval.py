# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import itertools
import os

import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.ref_reviser.dataset import tokenize


def process_example(fn):
    examples = pd.read_csv(fn)
    sent2perturb = dict(tuple(examples.groupby('sent_idx')))
    covs = []
    densities = []
    for sent_idx, sents in sent2perturb.items():
        example_id = sents['example_id'].tolist()[0]
        uid = f'{example_id}.{sent_idx}'
        if len(sents) == 1 or (uid_filter is not None and uid not in uid_filter):
            continue
        texts = sents['text'].tolist()
        a_toks, b_toks = tokenize(texts[0]), tokenize(texts[1])
        af = parse_extractive_fragments(a_toks, b_toks, remove_stop=True)
        bf = parse_extractive_fragments(b_toks, b_toks, remove_stop=True)
        covs.append((af['coverage'] + bf['coverage']) / 2.0)
        densities.append((af['density'] + bf['density']) / 2.0)
    return covs, densities


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate perturb outputs for diversity')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', required=True)
    parser.add_argument('-only_eval', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.5, type=float)

    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.target)

    free_gpus = get_free_gpus()

    noise_dir = os.path.join(data_dir, 'perturb', args.experiment, 'output')
    print(f'Reading in files from {noise_dir}')
    fns = glob(noise_dir + '/*.csv')
    print(f'Found {len(fns)} examples to process...')

    uid_filter = None
    if args.only_eval:
        # Only run evaluation code for the 1,000 set aside for evalution (just faster)
        uid_filter = set(pd.read_csv(os.path.join(data_dir, 'high_quality', 'eval_examples.csv'))['uid'])

    if args.cpu_frac == -1:
        outputs = list(tqdm(map(process_example, fns)))
    else:
        outputs = list((p_uimap(process_example, fns, num_cpus=args.cpu_frac)))

    covs = list(itertools.chain(*[x[0] for x in outputs]))
    densities = list(itertools.chain(*[x[1] for x in outputs]))

    mean_cov = np.mean(covs)
    mean_density = np.mean(densities)
    print(f'Mean intra-perturb unigram extractive coverage is {mean_cov}. Density={mean_density}')
    out_df = pd.DataFrame({'coverage': [mean_cov], 'density': [mean_density]})
    noise_dir = os.path.join(data_dir, 'perturb', args.experiment)
    eval_dir = os.path.join(noise_dir, 'diversity')
    os.makedirs(eval_dir, exist_ok=True)
    out_fn = os.path.join(eval_dir, 'extractive_frag.csv')
    print(f'Saving results to {out_fn}')
    out_df.to_csv(out_fn, index=False)
