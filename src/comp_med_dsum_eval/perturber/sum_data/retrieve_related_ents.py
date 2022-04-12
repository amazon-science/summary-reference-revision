# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import argparse
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap

from comp_med_dsum_eval.perturber.dsum_data.retrieve_related_ents import retrieve_ents


def process(i, all_neighbors, index_sents, sent_embed_dicts, top_k_ents=25):
    row = sent_embed_dicts[i]
    row.pop('vec')
    neighbors = all_neighbors[i]
    retrieved = [index_sents[neighbor_idx] for neighbor_idx in neighbors]
    retrieve_ents(row, retrieved, top_k_ents=top_k_ents)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Use index to retrieve entities from related sentences.')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('--top_k_sents', default=250, type=int)
    parser.add_argument('--top_k_ents', default=25, type=int)
    parser.add_argument('--mode', default='query', choices=['query', 'process'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=0, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    if args.debug:
        args.cpu_frac = -1

    if args.mode == 'query':
        index_fn = os.path.join(args.input_dir, 'dsum', 'sent_index.bin')
        print(f'Loading index at {index_fn}')
        index = faiss.read_index(index_fn)
        res = faiss.StandardGpuResources()
        print(f'Transfering index from CPU to cuda:{args.gpu_device}')
        index_gpu = faiss.index_cpu_to_gpu(res, args.gpu_device, index)

        embed_fn = os.path.join(args.input_dir, args.target, 'high_quality', f'sent_embeds{mini_str}.pk')
        print(f'Loading sentence embeddings at {embed_fn}')
        with open(embed_fn, 'rb') as fd:
            sent_embed_dicts = pickle.load(fd)

        sent_embeds = np.stack([x['vec'] for x in sent_embed_dicts])

        out_fn = os.path.join(args.input_dir, args.target, 'high_quality', f'nearest_neighbors{mini_str}.npy')
        chunksize = 100000
        num_chunks = max(1, len(sent_embeds) // chunksize)
        embed_chunks = np.array_split(sent_embeds, num_chunks)
        all_idxs = []
        for embed_chunk in tqdm(embed_chunks, total=len(embed_chunks)):
            idxs = index_gpu.search(embed_chunk, k=args.top_k_sents)[1]
            all_idxs.append(idxs)
        index_gpu.reset()
        all_idxs = np.concatenate(all_idxs)
        print(f'Saving sentence alignments to {out_fn}')
        np.save(out_fn, all_idxs)
    else:
        embed_fn = os.path.join(args.input_dir, args.target, 'high_quality', f'sent_embeds{mini_str}.pk')
        print(f'Loading sentence embeddings at {embed_fn}')
        with open(embed_fn, 'rb') as fd:
            sent_embed_dicts = pickle.load(fd)

        meta_fn = os.path.join(args.input_dir, 'dsum', 'sent_index_meta.csv')
        print(f'Loading sentence meta information at {meta_fn}')
        meta_df = pd.read_csv(meta_fn)
        index_sents = meta_df.to_dict('records')

        neighbor_fn = os.path.join(args.input_dir, args.target, 'high_quality', f'nearest_neighbors{mini_str}.npy')
        print(f'Loading retrievals from {neighbor_fn}')
        neighbors = np.load(neighbor_fn)
        n = len(sent_embed_dicts)
        assert len(sent_embed_dicts) == len(neighbors)
        if args.cpu_frac == -1:
            related_ents = pd.DataFrame(
                list(tqdm(map(
                    lambda i: process(i, neighbors, index_sents, sent_embed_dicts, args.top_k_ents),
                    range(n)), total=n
                )))
        else:
            related_ents = pd.DataFrame(list(p_uimap(
                lambda i: process(i, neighbors, index_sents, sent_embed_dicts, args.top_k_ents),
                range(n), num_cpus=args.cpu_frac
            )))

        meta_out_fn = os.path.join(args.input_dir, args.target, 'high_quality', f'sents_w_related_ents{mini_str}.csv')
        print(f'Saving {len(related_ents)} sentences to {meta_out_fn}')
        related_ents.to_csv(meta_out_fn, index=False)
