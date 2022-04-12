# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import os
import pickle

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
import regex as re
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForPreTraining

from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.eval.rouge import stopwords
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent


ELECTRA_FAKE_THRESHOLD = 0.25


def encode(text, model, tokenizer, device, max_length=128, top_n_layers=4, max_batch_size=100):
    seq_lens = []
    outputs = {'hidden_states': [], 'logits': []}
    text_batches = [list(x) for x in np.array_split(np.arange(len(text)), round(len(text) // max_batch_size) + 1)]
    text_batches = [x for x in text_batches if len(x) > 0]
    inputs = tokenizer(text, truncation=True, padding='longest', max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        for batch_idxs in text_batches:
            batch_inputs = {k: v[batch_idxs].to(device) for k, v in inputs.items()}
            seq_lens += batch_inputs['attention_mask'].sum(dim=1).tolist()
            batch_output = model(**batch_inputs, output_hidden_states=True)
            h_pool = torch.stack(batch_output['hidden_states'][-top_n_layers:]).mean(dim=0).cpu()
            outputs['hidden_states'].append(h_pool)
            if 'logits' in batch_output:
                logits = batch_output['logits'].cpu()
                outputs['logits'].append(logits)
    outputs['hidden_states'] = torch.cat(outputs['hidden_states'], dim=0)
    if len(outputs['logits']) > 0:
        outputs['logits'] = torch.cat(outputs['logits'], dim=0)
    return outputs, seq_lens


def bertscore_post_filt(a_h, b_h):
    if type(a_h) == torch.Tensor:
        a_h = a_h.cpu()
    if type(b_h) == torch.Tensor:
        b_h = b_h.cpu()
    sim_matrix = cosine_similarity(a_h, b_h)
    max_rows, max_cols = sim_matrix.max(axis=0), sim_matrix.max(axis=1)

    out_a, out_b = float(max_rows.mean()), float(max_cols.mean())
    f1 = (2 * out_a * out_b) / (out_a + out_b)
    return out_a, out_b, f1


def bertscore(a_tokens, a_h, b_tokens, b_h):
    a_keep = [i for i in range(min(len(a_tokens), len(a_h))) if a_tokens[i].lower() not in stopwords or
              (i < len(a_tokens) - 1 and a_tokens[i + 1].startswith('#'))]
    b_keep = [i for i in range(min(len(b_tokens), len(b_h))) if b_tokens[i].lower() not in stopwords or
              (i < len(b_tokens) - 1 and b_tokens[i + 1].startswith('#'))]

    a_h_keep = a_h[a_keep] if len(a_keep) > 0 else a_h
    b_h_keep = b_h[b_keep] if len(b_keep) > 0 else b_h

    return bertscore_post_filt(a_h_keep, b_h_keep)


def plausibility(tokens, logits):
    keep_idxs = [i for i in range(min(len(tokens), len(logits))) if tokens[i].lower() not in stopwords
                 and tokens[i] != '[UNK]']
    if len(keep_idxs) == 0:
        keep_idxs = [0]  # just to avoid error (not a big deal and unlikely to happen)
    fake_probs = torch.sigmoid(logits[keep_idxs])
    fake_predict = fake_probs > ELECTRA_FAKE_THRESHOLD
    most_fake = float(fake_probs.max())
    num_fake = int(fake_predict.sum())
    avg_fake = float(fake_probs.mean())
    return {
        'fake_score_max': most_fake,
        'fake_pred_num': num_fake,
        'fake_score_avg': avg_fake,
        'fake_score_candidate': len(fake_probs),
        'fake_frac': num_fake / max(1, len(fake_probs))
    }


def compute_bertscores(tokens, h, eval_outputs, prefix='bert'):
    curr_sent_h, curr_sent_tokens = None, None
    for i in range(len(eval_outputs)):
        if eval_outputs[i]['version'] == 'original':
            curr_sent_h = h[i]
            curr_sent_tokens = tokens[i]
        else:
            perturb_h = h[i]
            perturb_tokens = tokens[i]
            bs_recall, bs_precision, _ = bertscore(perturb_tokens, perturb_h, curr_sent_tokens, curr_sent_h)
            eval_outputs[i][f'{prefix}_bs_recall'] = bs_recall
            eval_outputs[i][f'{prefix}_bs_precision'] = bs_precision


def process_example(
        out_dir, fn, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device, top_n_layers=4,
        uid_filter=None
):
    example_id = fn.split('/')[-1].replace('.csv', '')
    examples = pd.read_csv(fn)
    records = examples.to_dict('records')
    texts = []
    eval_outputs = []
    for i, record in enumerate(records):
        sent_idx = record['sent_idx']
        uid = f'{example_id}.{sent_idx}'
        if uid_filter is not None and uid not in uid_filter:
            continue

        perturb_idx = record['perturb_idx']
        sent_key = str(sent_idx)
        if i == 0 or str(sent_idx) != str(records[i - 1]['sent_idx']):
            texts.append(record['text_original'])
            eval_outputs.append(
                {'example_id': example_id, 'key': sent_key, 'sent_idx': sent_idx, 'version': 'original',
                 'perturb_idx': None}
            )

        perturb_key = f'{sent_idx}_{perturb_idx}'
        texts.append(record['text'])
        try:
            span_remove = int(re.search(r'<span-remove-(\d+)>', record['masked_str']).group(1))
            ent_add = int(re.search(r'<ent-add-(\d+)>', record['masked_str']).group(1))
            ent_remove = int(re.search(r'<ent-remove-(\d+)>', record['masked_str']).group(1))
            shuffle = int(re.search(r'<shuffle-(\d+)>', record['masked_str']).group(1))
        except:
            span_remove = ent_add = ent_remove = shuffle = None

        row = {
            'example_id': example_id, 'key': perturb_key, 'sent_idx': sent_idx, 'version': 'perturbed',
            'perturb_idx': perturb_idx, 'span_remove': span_remove, 'ent_add': ent_add, 'ent_remove': ent_remove,
            'shuffle_orderliness': shuffle
        }
        eval_outputs.append(row)

    if len(eval_outputs) == 0:
        return

    texts = list(map(remove_tags_from_sent, texts))

    electra_tokens = list(map(electra_tokenizer.tokenize, texts))
    electra_outputs, electra_seq_lens = encode(
        texts, electra_model, electra_tokenizer, device=device, max_length=256, top_n_layers=top_n_layers)
    # remove the CLS and final </s>
    electra_pooled_h = electra_outputs['hidden_states'][:, 1:-1]
    compute_bertscores(electra_tokens, electra_pooled_h, eval_outputs, prefix='electra')

    fake_metrics = [plausibility(
        electra_tokens[i], electra_outputs['logits'][i, 1:-1]) for i in range(len(electra_tokens))]
    for i in range(len(fake_metrics)):
        eval_outputs[i].update(fake_metrics[i])

    bert_tokens = list(map(clinbert_tokenizer.tokenize, texts))
    bert_outputs, bert_seq_lens = encode(
        texts, clinbert_model, clinbert_tokenizer, device, max_length=128, top_n_layers=top_n_layers)
    # remove the CLS and final </s>
    bert_pooled_h = bert_outputs['hidden_states'][:, 1:-1]
    compute_bertscores(bert_tokens, bert_pooled_h, eval_outputs, prefix='bert')
    out_fn = os.path.join(out_dir, f'{example_id}.csv')
    out_df = pd.DataFrame(eval_outputs)
    out_df.to_csv(out_fn, index=False)

    bert_outputs['hidden_states'] = bert_outputs['hidden_states'].numpy()
    electra_outputs['hidden_states'] = electra_outputs['hidden_states'].numpy()
    electra_outputs['logits'] = electra_outputs['logits'].numpy()

    expanded_outputs = []
    for i, record in enumerate(eval_outputs):
        record['bert_hidden_states'] = bert_outputs['hidden_states'][i, :bert_seq_lens[i], :]
        record['bert_tokens'] = bert_tokens[i]
        record['electra_hidden_states'] = electra_outputs['hidden_states'][i, :electra_seq_lens[i], :]
        record['electra_logits'] = electra_outputs['logits'][i][:electra_seq_lens[i]]
        record['electra_tokens'] = electra_tokens[i]
        expanded_outputs.append(record)
    out_fn = os.path.join(out_dir, f'{example_id}.pk')
    with open(out_fn, 'wb') as fd:
        pickle.dump(expanded_outputs, fd)


def process_chunk(
        out_dir, fns, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device, uid_filter=None):
    electra_model = electra_model.to(device)
    clinbert_model = clinbert_model.to(device)
    for fn in tqdm(fns, total=len(fns)):
        process_example(
            out_dir, fn, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device,
            uid_filter=uid_filter
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate perturb outputs for plausibility and semantic variance')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--experiment', default='ent_sample')
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-only_eval', default=False, action='store_true')
    parser.add_argument('-only_new', default=False, action='store_true')
    parser.add_argument(
        '--hf_model', default='sultan/BioM-ELECTRA-Large-Discriminator',
        choices=[
            'kamalkraj/bioelectra-base-discriminator-pubmed',
            'sultan/BioM-ELECTRA-Large-Discriminator'
        ])
    parser.add_argument('--num_gpus', default=1, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''

    if args.debug:
        args.cpu_frac = -1
        args.num_gpus = 1

    data_dir = os.path.join(args.input_dir, args.target)
    noise_dir = os.path.join(data_dir, 'perturb', args.experiment, 'output')
    fns = glob(noise_dir + '/*.csv')
    print(f'Found {len(fns)} examples to process...')

    uid_filter = None
    if args.only_eval:
        # Only run evaluation code for the 1,000 set aside for evaluation (just faster)
        uid_filter = set(pd.read_csv(os.path.join(data_dir, 'high_quality', 'eval_examples.csv'))['uid'])

    out_dir = os.path.join(data_dir, 'perturb', args.experiment, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    if args.only_new:
        example_ids = [x.split('/')[-1].replace('.csv', '') for x in fns]
        meta_pattern = out_dir + '/*.csv'
        meta_ent_fns = glob(meta_pattern)
        done_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in meta_ent_fns])
        print(f'Choosing not to re-evaluate for the {len(done_example_ids)} already completed examples')
        fns = [fn for fn, example_id in zip(fns, example_ids) if example_id not in done_example_ids]

    fn_chunks = np.array_split(fns, args.num_gpus)
    config = AutoConfig.from_pretrained(args.hf_model)

    electra_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    electra_model = AutoModelForPreTraining.from_pretrained(args.hf_model)

    clinbert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    clinbert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    free_gpus = get_free_gpus()

    if args.num_gpus == 1:
        process_chunk(
            out_dir, fns, electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, free_gpus[0],
            uid_filter=uid_filter
        )
    else:
        free_gpus = free_gpus[:len(fn_chunks)]
        list(p_uimap(lambda x: process_chunk(
            out_dir, x[0], electra_model, electra_tokenizer, clinbert_model, clinbert_tokenizer, device=x[1],
            uid_filter=uid_filter
        ), list(zip(fn_chunks, free_gpus)), num_cpus=len(fn_chunks)))
