# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
import itertools
import os
import ujson
import regex as re

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from comp_med_dsum_eval.preprocess.sec_tag.section_utils import get_attr
from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments
from comp_med_dsum_eval.ref_reviser.dataset import tokenize, remove_tags_from_sent
from comp_med_dsum_eval.preprocess.entity.process_ents import add_ent_id, filter_ents
from comp_med_dsum_eval.gen_transformers.dataset import add_meta


def min_max(df, col):
    return MinMaxScaler().fit_transform(np.array(df[col]).reshape(-1, 1))


def first_context_sent(revised_df):
    context = revised_df.context.tolist()[0]
    first_sent = context.split('</s>')[0]
    source_sent_idx = re.findall('idx=(\d+)', first_sent)[0]
    # num_ents = len(re.findall('</e>', first_sent))
    context_str = remove_tags_from_sent(first_sent)
    # Fix this
    row = {
        'prediction': context_str,
        # 'num_ents': num_ents,
        'source_sent_idx': source_sent_idx,
        # 'global_halluc': 0,
        'source_extract_code': None,
        'input_extract_code': None,
    }
    return row


def replace_sent_max_coverage(revised_df):
    row = revised_df.iloc[revised_df.gen_context_cov_improve.argmax()]
    return row


def replace_sent_balanced(revised_df, verbose=False):
    revised_df['gen_context_cov_improve_scaled'] = min_max(revised_df, 'gen_context_cov_improve')
    revised_df['gen_input_sim_scaled'] = min_max(revised_df, 'gen_input_sim')
    revised_df['source_density_scaled'] = min_max(revised_df, 'source_density')

    records = revised_df.to_dict('records')
    scores = []
    for record in records:
        inputs = np.array([
            record['gen_context_cov_improve_scaled'],
            # record['gen_input_sim_scaled'],
            1 - record['source_density_scaled']  # we want less copy-paste, preferably
        ]) + 1e-4
        score = np.average(inputs, weights=[2, 1])
        scores.append(score)
    best_score = int(np.argmax(scores))
    best = records[best_score]
    if verbose:
        other_prediction = revised_df.iloc[revised_df.gen_context_cov_improve.argmax()].prediction
        target_sent = revised_df.target_sent.tolist()[0]
        context = revised_df.context.tolist()[0]
        context_str = remove_tags_from_sent(' '.join(context.split('</s>')))
        target_str = remove_tags_from_sent(target_sent)
        print('Context: ', context_str)
        print('Target: ', target_str)
        print('Prediction: ', best['prediction'])
        print(other_prediction)
        print(best['source_extract_code'])
        print(best['gen_context_cov_improve'], revised_df.gen_context_cov_improve.max())
        print(best['source_density'], revised_df.source_density.max())
        print('\n')
    return best


def process(record, data_dir, revise_experiment, replace_func):
    example_id = record['example_id']
    generated_fn = os.path.join(data_dir, 'revise', 'output', revise_experiment, f'{example_id}.csv')
    sentidx2revised = {}
    gen_eval_df = None
    revise_target_merges = revise_source_merges = []
    if os.path.exists(generated_fn):
        gen_eval_fn = os.path.join(data_dir, 'revise', 'eval', revise_experiment, f'{example_id}.csv')
        gen_eval_df = pd.read_csv(gen_eval_fn)
        gen_str = pd.read_csv(generated_fn)
        sentidx2revised = dict(tuple(gen_str.groupby(by='target_sent_idx')))
    # [check] extractive: coverage, density
    # Entity: number of entities, hallucination rate, relevance fraction
    # for revised sentences -> ent_eval_df
    # Coherence
    orig_context_fn = os.path.join(data_dir, 'revise', 'contexts', f'{example_id}.json')
    with open(orig_context_fn, 'r') as fd:
        orig_contexts = ujson.load(fd)

    # Recomment if you want to run these evaluations on the ent annotated versions
    # ent_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.json')
    # with open(ent_fn) as fd:
    #     ents = ujson.load(fd)
    # source_ents = ents['source']
    # ents_flat = list(itertools.chain(
    #     *[filter_ents(add_ent_id(ent_obj, 'source')) for ent_obj in source_ents]
    # ))
    # all_source_ent_ids = [x['ent_id'] for x in ents_flat]
    # try:
    #     orig_merge_fn = os.path.join(data_dir, 'acm_output', f'{example_id}.csv')
    #     orig_merges = pd.read_csv(orig_merge_fn)
    #     orig_merges = orig_merges[orig_merges['should_merge']].dropna(
    #         subset=['source_ent_id', 'target_ent_id']).to_dict('records')
    # except:
    #     orig_merges = []

    target = record['target']
    target_tps = target.split('<SEP>')
    target_sent_idx = None
    modified_tps = []
    # covered_source_ent_ids = set()
    # covered_target_ent_ids = set()
    sent_level_stats = []
    orig_stats = None
    for tp_idx, tp in enumerate(target_tps):
        if tp.startswith('<s'):
            target_sent_idx = int(get_attr(tp, 'idx'))
            try:
                orig_stats = [x for x in orig_contexts if x['target']['sent_idx'] == target_sent_idx][0]['stats']
            except:
                print(f'No retrievals for target sentence {target_sent_idx} in {example_id} likely because too long.')
                # TODO (low priority)
                # Ensure we do retrievals for every sentence in rev_reviser.alignments.retrieve_contexts
                orig_stats = {}
        elif tp_idx > 0 and target_tps[tp_idx - 1].startswith('<s') and target_sent_idx in sentidx2revised:
            row = replace_func(sentidx2revised[target_sent_idx])
            source_extract_code = row['source_extract_code']
            if source_extract_code is not None:  # extraction
                bert_overlap = gen_eval_df[
                    (gen_eval_df['target_sent_idx'] == target_sent_idx) &
                    (gen_eval_df['source_extract_code'] == source_extract_code)
                ]
                assert len(bert_overlap) == 1
                bert_overlap = bert_overlap.to_dict('records')[0]
                # ent_id_frag = f'sent-{target_sent_idx}-revise-{source_extract_code}-'
                # for ent_merge in revise_source_merges:
                #     if ent_id_frag in ent_merge['revised_ent_id']:
                #         covered_source_ent_ids.add(ent_merge['source_ent_id'])
                # for ent_merge in revise_target_merges:
                #     if ent_id_frag in ent_merge['revised_ent_id']:
                #         covered_target_ent_ids.add(ent_merge['target_ent_id'])
            else:
                bert_overlap = {
                    'bert_bs_con_cov': 1,
                    'bert_bs_con_prec': 1,
                    'bert_bs_con_f1': 1,
                }
                source_sent_idx = row['source_sent_idx']
                # source_sent_id_frag = f'sent-{source_sent_idx}-'
                # for st_merges in orig_merges:
                #     if source_sent_id_frag in st_merges['source_ent_id']:
                #         covered_source_ent_ids.add(st_merges['source_ent_id'])
                #         covered_target_ent_ids.add(st_merges['target_ent_id'])
            sent_stats = {
                'example_id': example_id,
                'target_sent_idx': target_sent_idx,
                'revised': True,
                # 'global_halluc': row['global_halluc'],
                'source_extract_code': source_extract_code,
                # 'num_ents': row['num_ents'],
            }
            sent_stats.update(bert_overlap)
            sent_level_stats.append(sent_stats)
            # Replace with prediction
            tp = row['prediction']
        elif tp_idx > 0 and target_tps[tp_idx - 1].startswith('<s'):
            # non-revised sentence
            sent_stats = {
                'example_id': example_id,
                'target_sent_idx': target_sent_idx,
                'revised': False,
                # 'global_halluc': orig_stats.get('num_hallucinations', None),
                'source_extract_code': None,
                'num_ents': orig_stats.get('target_nums', None),
                'bert_bs_con_cov': orig_stats.get('source_to_target_coverage', None),
                'bert_bs_con_prec': orig_stats.get('target_to_source_coverage', None)
            }
            try:
                sent_stats['bert_bs_con_f1'] = (2 * sent_stats['bert_bs_con_cov'] * sent_stats['bert_bs_con_prec']) / (
                        sent_stats['bert_bs_con_cov'] + sent_stats['bert_bs_con_prec']
                )
            except:
                sent_stats['bert_bs_con_f1'] = None

            sent_level_stats.append(sent_stats)
            # for st_merges in orig_merges:
            #     target_ent_id_frag = f'sent-{target_sent_idx}-'
            #     if target_ent_id_frag in st_merges['target_ent_id']:
            #         covered_source_ent_ids.add(st_merges['source_ent_id'])
            #         covered_target_ent_ids.add(st_merges['target_ent_id'])

        modified_tps.append(tp)
    record['target'] = '<SEP>'.join(modified_tps)
    record['original_target'] = target
    # faithful_orig_target_ents = set([x['target_ent_id'] for x in orig_merges])
    # faithful_target_ents = len(faithful_orig_target_ents)
    # covered_faithful_target_ents = len(covered_target_ent_ids.intersection(faithful_orig_target_ents))

    source_toks = tokenize(remove_tags_from_sent(''.join(record['source'].split('<SEP>'))))
    target_toks = tokenize(remove_tags_from_sent(''.join(record['target'].split('<SEP>'))))
    frags = parse_extractive_fragments(source_toks, target_toks, remove_stop=True)
    record['pred_target_toks'] = len(target_toks)
    sent_stats_df = pd.DataFrame(sent_level_stats)
    if len(sent_level_stats) > 0:
        record['num_revised'] = sent_stats_df['revised'].sum()
        record['num_sents'] = len(sent_stats_df)
        # record['global_halluc'] = sent_stats_df['global_halluc'].dropna().sum()
        # record['num_ents'] = sent_stats_df['num_ents'].dropna().sum()
        record['avg_bert_bs_con_cov'] = sent_stats_df['bert_bs_con_cov'].dropna().mean()
        if 'input_extract_code' in list(sent_stats_df.columns):
            record['avg_input_extract_code'] = sent_stats_df['input_extract_code'].dropna().mean()
            record['avg_source_extract_code'] = sent_stats_df['source_extract_code'].dropna().mean()
        else:
            record['avg_input_extract_code'] = record['avg_source_extract_code'] = None
    else:
        record['num_revised'] = 0
        record['num_sents'] = 0
        # record['global_halluc'] = 0
        # record['num_ents'] = 0
        record['avg_input_extract_code'] = None
        record['avg_source_extract_code'] = None
        record['avg_bert_bs_con_cov'] = 1
    # record['halluc_frac'] = 0 if record['num_ents'] == 0 else record['global_halluc'] / record['num_ents']
    # denom = len(all_source_ent_ids)
    # record['source_ent_cov'] = 1 if denom == 0 else len(covered_source_ent_ids) / denom
    # record['covered_faithful_target_ents'] = covered_faithful_target_ents
    # record['faithful_target_ents'] = faithful_target_ents
    # record['ent_rel_frac'] = 1 if faithful_target_ents == 0 else covered_faithful_target_ents / faithful_target_ents
    record.update(frags)
    return record, sent_level_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Build improved summaries by inserting revised sentences for low quality sentences.'
    )
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('--revise_experiment', default='yay')
    parser.add_argument('--max_n', default=None, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    # some summaries have 0 sentences that need revision.
    # If you don't put this flag to true, it will include all summaries
    parser.add_argument('-just_revised', default=False, action='store_true')
    parser.add_argument('--replace_strategy', default='balanced', choices=[
        'balanced', 'max_coverage', 'extractive'
    ])

    args = parser.parse_args()
    if args.debug:
        args.cpu_frac = -1

    data_dir = os.path.join(args.input_dir, args.target)

    mini_str = '_mini' if args.debug else ''
    in_fn = os.path.join(data_dir, f'summary_dataset{mini_str}.csv')
    print(f'Reading in data from {in_fn}')
    summary_df = pd.read_csv(in_fn)
    summary_df = add_meta(summary_df, data_dir)
    summary_df = summary_df[summary_df['split'] == 'train']
    if args.max_n is not None and args.max_n < len(summary_df):
        print(f'Randomly shrinking dataset to {args.max_n}')
        summary_df = summary_df.sample(n=args.max_n, replace=False, random_state=1992)

    if args.just_revised:
        revise_dir = os.path.join(data_dir, 'revise', 'output', args.revise_experiment)
        generated_fns = glob(revise_dir + '/*.csv')
        generated_example_ids = set([x.split('/')[-1].replace('.csv', '') for x in generated_fns])
        summary_df = summary_df[summary_df['example_id'].isin(generated_example_ids)]
    summary_df = summary_df.sort_values(by='source', key=lambda x: x.str.len())
    records = summary_df.to_dict('records')

    if args.replace_strategy == 'extractive':
        replace_func = first_context_sent
    elif args.replace_strategy == 'max_coverage':
        replace_func = replace_sent_max_coverage
    elif args.replace_strategy == 'balanced':
        replace_func = replace_sent_balanced
    else:
        raise Exception(f'Unrecognized sentence strategy --> {args.replace_strategy}')

    if args.cpu_frac == -1:
        outputs = list(tqdm(map(lambda record: process(
            record, data_dir, args.revise_experiment, replace_func), records), total=len(records)))
    else:
        outputs = list(p_uimap(lambda record: process(
            record, data_dir, args.revise_experiment, replace_func), records))

    out_df = pd.DataFrame([x[0] for x in outputs])
    sent_level_stats = pd.DataFrame(list(itertools.chain(*[x[1] for x in outputs])))

    max_n_str = '' if args.max_n is None else '_' + str(args.max_n)
    out_fn = os.path.join(
        data_dir, f'summary_dataset_{args.revise_experiment}_revised_{args.replace_strategy}{mini_str}{max_n_str}.csv'
    )
    print(f'Saving {len(out_df)} examples to {out_fn}')
    out_df.to_csv(out_fn, index=False)

    stats_fn = os.path.join(
        data_dir, f'stats_for_summary_dataset_{args.replace_strategy}_revised{mini_str}{max_n_str}.csv'
    )
    print(f'Saving {len(out_df)} sentence-level statistics to {stats_fn}')
    sent_level_stats.to_csv(stats_fn, index=False)

    print('Number of tokens per revised reference: ', out_df['pred_target_toks'].dropna().mean())
    # print('Num entities: ', out_df['num_ents'].mean())
    print('Average extractive coverage: ', out_df['coverage'].mean())
    print('Average extractive density: ', out_df['density'].mean())
    # print('Hallucination fraction: ', out_df['halluc_frac'].mean())
    print('Average BertScore coverage: ', out_df['avg_bert_bs_con_cov'].mean())
    # print('Source Entity Coverage: ', out_df['source_ent_cov'].mean())
    # print('Faithful adjusted recall: ', out_df['ent_rel_frac'].mean())

    if args.replace_strategy != 'extractive':
        print('Average Input Extract Code: ', out_df['avg_input_extract_code'].dropna().mean())
        print('Average Source Extract Code: ', out_df['avg_source_extract_code'].dropna().mean())
