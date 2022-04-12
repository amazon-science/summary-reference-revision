# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import argparse
import pickle
import pandas as pd
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from comp_med_dsum_eval.gpu_utils import get_free_gpus
from comp_med_dsum_eval.preprocess.sec_tag.main import section_tagger_init
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import sents_from_html, get_sent_idxs
from comp_med_dsum_eval.perturber.evaluate import encode
from comp_med_dsum_eval.preprocess.variations import mapping_variations
from comp_med_dsum_eval.preprocess.structure_notes import tag_text
from comp_med_dsum_eval.ref_reviser.alignments.utils import keep_sent
from comp_med_dsum_eval.ref_reviser.alignments.retrieve_contexts import retrieve_context
from comp_med_dsum_eval.ref_reviser.build_train_dataset_from_perturbed import add_to_embeds


def process(example, bert_model, bert_tokenizer, source_html, gpu_device=0):
    example_id = example['example_id']

    source_sents = sents_from_html(source_html)
    source_idxs = get_sent_idxs(source_html)
    assert len(source_idxs) == len(source_sents)

    used_sents = set()
    used_sent_map = {}
    for sent_idx, source_sent in zip(source_idxs, source_sents):
        if source_sent.lower() in used_sents:
            continue
        used_sents.add(source_sent.lower())
        used_sent_map[sent_idx] = source_sent
    used_sent_idxs = set(used_sent_map.keys())
    prediction_tagged = example['prediction_tagged']
    predicted_sents = sents_from_html(prediction_tagged)
    n = len(predicted_sents)

    embed_fn = os.path.join(data_dir, 'embed_cache', f'{example_id}.pk')
    with open(embed_fn, 'rb') as fd:
        embeds = pickle.load(fd)

    source_embeds = [embed for embed in embeds['source'] if embed['sent_idx'] in used_sent_idxs]
    bert_tokens = list(map(bert_tokenizer.tokenize, predicted_sents))
    bert_outputs, bert_seq_lens = encode(predicted_sents, bert_model, bert_tokenizer, gpu_device)
    outputs = []
    for i in range(n):
        bert_h = bert_outputs['hidden_states'][i, :bert_seq_lens[i]].numpy()
        row = {'sent_idx': i, 'bert_token': bert_tokens[i], 'bert_h': bert_h}
        add_to_embeds(row, tok_col='bert_token', h_col='bert_h')
        added_sents, added_sent_idxs, max_coverages, added_scores, added_priors, improvements = retrieve_context(
            used_sent_map, source_embeds, row)
        kept_sents = [added_sents[add_idx] for add_idx in range(len(added_sents)) if keep_sent(improvements[add_idx])]
        output_row = {
            'example_id': example_id,
            'predicted_sent_idx': i,
            'prediction': predicted_sents[i],
            'context': '<s>'.join(kept_sents)
        }
        outputs.append(output_row)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to retrieve contexts for model predictions')
    parser.add_argument('--input_dir', default='/efs/griadams')
    parser.add_argument('--experiment', default='longformer_16384_full')
    parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)

    args = parser.parse_args()
    mini_str = '_mini' if args.debug else ''
    if args.debug:
        args.cpu_frac = -1

    free_gpus = get_free_gpus()
    args.gpu_device = free_gpus[0] if args.gpu_device is None else args.gpu_device
    # if args.gpu_device is not None and args.gpu_device not in free_gpus:
    #     print(f'Warning! You\'ve selected a GPU that is not available.  Putting the model on {free_gpus[0]} instead.')
    #     args.gpu_device = free_gpus[0]

    section_tagger_init()
    print('Loading Spacy...')
    sentencizer = spacy.load('en_core_sci_sm', disable=['ner', 'parser', 'lemmatizer'])
    sentencizer.add_pipe('sentencizer')

    variation = 'history_of_present_illness' if args.target == 'hpi' else 'brief_hospital_course'
    target_headers = [x[0] for x in mapping_variations[variation]]

    data_dir = os.path.join(args.input_dir, args.target)
    experiment_dir = os.path.join(data_dir, 'mimic_sum', 'results', args.experiment)
    gen_fn = os.path.join(experiment_dir, 'outputs_annotated.csv')  # TODO change to just outputs.csv
    gen_df = pd.read_csv(gen_fn)
    gen_df = gen_df.assign(
        prediction_tagged=gen_df['prediction'].apply(
            lambda text: '<SEP>'.join(tag_text(text, sentencizer, prepend_sub_sections=True))
        )
    )
    records = gen_df.to_dict('records')

    bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').eval().to(args.gpu_device)

    fn = os.path.join(data_dir, f'summary_dataset_rouge_annotated{mini_str}.csv')
    print(f'Reading dataset from {fn}')
    full_examples = pd.read_csv(fn)
    example2source = dict(zip(full_examples['example_id'], full_examples['source']))

    outputs = pd.DataFrame(list(itertools.chain(*list(tqdm(map(lambda record: process(
        record, bert_model, bert_tokenizer, source_html=example2source[record['example_id']],
        gpu_device=args.gpu_device
    ), records), total=len(records))))))
    out_fn = os.path.join(experiment_dir, 'contexts.csv')
    print(f'Saving {len(outputs)} contexts for {len(gen_df)} predictions to {out_fn}')
    outputs.to_csv(out_fn, index=False)
