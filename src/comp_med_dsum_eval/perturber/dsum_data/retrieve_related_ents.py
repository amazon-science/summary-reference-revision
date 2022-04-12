# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import os
import pickle
import regex as re

import argparse
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap

from comp_med_dsum_eval.preprocess.constants import HTML_REGEX
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import get_attr
from comp_med_dsum_eval.preprocess.entity.entity_utils import extract_ents


ENT_ORDER = ['dx', 'procedure', 'treatment', 'test', 'med']
ENT_TYPE_MAP = {
    'DX_NAME': 'dx',
    'PROCEDURE_NAME': 'procedure',
    'TREATMENT_NAME': 'treatment',
    'TEST_NAME': 'test',
    'BRAND_NAME': 'med',
    'GENERIC_NAME': 'med'
}


def retrieve_ents(row, retrieved, top_k_ents=25):
    _, _, _, _, ents_in_sent = extract_ents(row['text'], set())
    curr_ent_set = set([x.lower() for x in ents_in_sent])
    retrieved_texts = [x['text'] for x in retrieved]
    processed_ents = set()
    ents_by_type = defaultdict(list)
    ranks_by_type = defaultdict(list)
    for sent_rank, text in enumerate(retrieved_texts):
        tps = re.split(HTML_REGEX, text)
        for tp_idx, tp in enumerate(tps):
            if tp == '</e>':
                ent_str = tps[tp_idx - 1]
                ent_type = get_attr(tps[tp_idx - 2], 'type')
                ent_type_resolved = ENT_TYPE_MAP[ent_type]
                is_new_ent = ent_str.lower() not in processed_ents and ent_str.lower() not in curr_ent_set
                is_addable = len(ents_by_type[ent_type_resolved]) < top_k_ents
                if is_new_ent and is_addable:
                    ents_by_type[ent_type_resolved].append(ent_str)
                    ranks_by_type[ent_type_resolved].append(str(sent_rank + 1))
                    processed_ents.add(ent_str.lower())

    num_related_ents = 0
    for type, ents in ents_by_type.items():
        row[type] = '<SEP>'.join(ents)
        num_related_ents += len(ents)
        row['num_' + type] = len(ents)
    row['num_related_ents'] = num_related_ents


def process(i, all_neighbors, records, top_k_ents=25):
    row = records[i]
    neighbors = all_neighbors[i]
    retrieved = [records[neighbor_idx] for neighbor_idx in neighbors]
    retrieve_ents(row, retrieved, top_k_ents=top_k_ents)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Use index to retrieve entities from related sentences.')
    parser.add_argument('--input_dir', default='/efs/griadams/dsum')
    parser.add_argument('--cpu_frac', default=0.75, type=float)
    parser.add_argument('--top_k_sents', default=250, type=int)
    parser.add_argument('--top_k_ents', default=25, type=int)
    parser.add_argument('--mode', default='query', choices=['query', 'process'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--max_n', default=None, type=int)

    args = parser.parse_args()

    if args.debug:
        args.cpu_frac = -1

    meta_fn = os.path.join(args.input_dir, 'sent_index_meta.csv')
    print(f'Loading sentence meta information at {meta_fn}')
    meta_df = pd.read_csv(meta_fn)

    if args.mode == 'query':
        index_fn = os.path.join(args.input_dir, 'sent_index.bin')
        print(f'Loading index at {index_fn}')
        index = faiss.read_index(index_fn)
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, args.gpu_device, index)

        embed_fn = os.path.join(args.input_dir, 'sent_embeds.pk')
        print(f'Loading sentence embeddings at {embed_fn}')
        with open(embed_fn, 'rb') as fd:
            sent_embeds = pickle.load(fd)
        faiss_indices = meta_df.index.tolist()

        out_fn = os.path.join(args.input_dir, 'nearest_neighbors.npy')
        chunksize = 100000
        num_chunks = len(sent_embeds) // chunksize
        embed_chunks = np.array_split(sent_embeds, num_chunks)
        all_idxs = []
        for embed_chunk in tqdm(embed_chunks, total=len(embed_chunks)):
            idxs = index_gpu.search(embed_chunk, k=args.top_k_sents)[1]
            all_idxs.append(idxs)
        index_gpu.reset()
        all_idxs = np.concatenate(all_idxs)
        np.save(out_fn, all_idxs)
    else:
        records = meta_df.to_dict('records')
        neighbor_fn = os.path.join(args.input_dir, 'nearest_neighbors.npy')
        print(f'Loading retrievals from {neighbor_fn}')
        neighbors = np.load(neighbor_fn)
        n = len(records)
        if args.max_n is not None and args.max_n < n:
            print(f'Only retrieving for first {args.max_n}.  Useful for debugging.')
            n = args.max_n
        assert len(records) == len(neighbors)
        if args.cpu_frac == -1:
            related_ents = pd.DataFrame(list(tqdm(map(lambda i: process(
                i, neighbors, records, args.top_k_ents), range(n)), total=n)))
        else:
            related_ents = pd.DataFrame(list(p_uimap(
                lambda i: process(i, neighbors, records, args.top_k_ents),
                range(n), num_cpus=args.cpu_frac
            )))

        meta_out_fn = os.path.join(args.input_dir, f'sent_index_meta_w_related_ents.csv')
        print(f'Saving {len(related_ents)} sentences to {meta_out_fn}')
        related_ents.to_csv(meta_out_fn, index=False)

        mini_out_fn = os.path.join(args.input_dir, f'sent_index_meta_w_related_ents_mini.csv')
        related_ents_mini = related_ents.sample(n=128, replace=False, random_state=1992)
        print(f'Saving {len(related_ents_mini)} sentences to {mini_out_fn}')
        related_ents_mini.to_csv(mini_out_fn, index=False)
