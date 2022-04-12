# Script Order

xx

## Generating Alignments (`./alignments`)

**Script Order**

1. `retrieve_contexts.py`
2. `collect_examples.py`
3. `collect_sentences.py`
4. `separate_alignments_by_quality.py`

Use `-only_new` to avoid repeating already done examples.

## retrieve_contexts

For each sentence in the dataset, retrieve the top 5 closest source sentences with BERTScore with coverage and adds additional sentences based on missing entities.

For each reference sentence, output is:

```angular2html
dataset_output = {
    'example_id': example_id,
    'source': {
        'sent_idxs': added_sent_idxs,
        'num_tokens': num_source_toks,
        'alignment': source_alignments,
        'sents': annotated_source,
        'improvements': improvements,
    }, 'target': {
        'alignment': target_alignments,
        'sent_idx': target_sent_idx,
        'previous_sent': '' if target_sent_idx == 0 else target_sents[target_sent_idx],
        'sent': target_sent_annotated,
    }, 'stats': stats
}
```

**source**
- added_sent_idxs = [list] source sent idxs of added sentences
- num_tokens = [list] number of tokens for each added sentence
- alignment = {dict} information regarding BERT embeddings & alignments [probably won’t be needed but it is cached
here]
sents = annotated source text (annotated with entities)
improvements = see paper (mean and max bertscore coverage improvement over previous extractions). we will use
these to filter out source sentences that don’t add much new relevant information (used in next two scripts to filter)

**target**

- alignment = {dict} information regarding BERT embeddings & alignments [probably won’t be needed but it is cached
here]
- sent_idx = target sent index
- previous_sent = this is unused but it’s just the previous reference sentence (for coherence analysis with next
sentence prediction. You can also get the previous sentence by loading summary_dataset.csv and and getting
sent_idx - 1 in ‘target’)

**stats**

```angular2html
stats = {
   'target_ent_num': number of ents in the target sentence,
   'source_linked_target_ent_num': number of faithful target ents,
   'num_hallucinations': number of hallucinated entities (target_ent_num - source_lin
   'covered_ent_num': total_covered_ents,
   'source_sents': len(added_sent_idxs),
   'source_to_target_coverage': BERTScore Precision (how well covered each target tok
   'target_to_source_coverage': BERTScore Recall(how well covered each target token i
   'source_toks': num_source_toks,
   'target_nums': number of numbers in target (not used but could be if you want to a
   'covered_nums': Number of faithful numbers (i.e., 5mg) -- found in both source con
}
```

Each example is dumped to a file under `{input_dir}/bhc/contexts/{example_id}.json` and each is a list of
the target reference sentences. There is one quirk that doesn’t occur often but could be addressed. If the sentence is too
long (exceeds BERT 128 tokens, we skip the sentence and don’t align it. This is mostly just a code issue where it will break
based on the tokenization being longer than the hidden state sequence — which is max 128). You can just revisit this by
tokenizing to a maximum of 128 tokens but it doesn’t affect many sentences. Just important to note if you find a small
subset of missing retrieved contexts.

## collect_examples, collect_sentences

These functions just collect all the reference sentence-source alignments  from `{input_dir}/bhc/contexts/{example_id}.json` and flatten them into a single data item. I probably didn’t
have to do this twice but you need to run both because both output formats are consumed by different models. 

- During collection, we filter out sentences that don’t meet the improvements threshold.
- `collect_examples` computes the sent_quality as `high`, `mid`, `low`, or `high_w_ent`. `high_w_ent` means no hallucinations and
`BERTScore >= 0.75` AND nonzero entities. `high` (not `high_w_ent`) includes sentences with no entities and thus no
hallucinations. 
- We train the reviser on `high_w_ent
separate_alignments_by_quality`.

## separate_alignments_by_quality

This is similar to collect with the exception that we filter for a specific quality cohort (low, mid, high). In this file, we also
randomly sample a ref reviser evaluation cohort that is a subset of the generation data (just quicker to compute).

# Training the Reviser

## main.py

The training data is built from `reviser/alignments/collect_sents.py` and is located at `{input_dir}/revise/dataset.json`.

Ablation Hyper-parameters shown below:

```angular2html
parser.add_argument('-remove_contrast', default=False, action='store_true')
parser.add_argument('-remove_mask', default=False, action='store_true')
parser.add_argument('-remove_neg', default=False, action='store_true')
   parser.add_argument('--contrast_input_strategy', default='worst', choices=['worst', 'best', 'random']
```

and from `parser = TransformerReviser.add_model_specific_args(parser)`

```
parser.add_argument('-from_perturb_checkpoint', default=False, action='store_true')
```

- `remove_contrast`: this is simple. only train on the un-masking objective. This isn’t shown in the paper because it doesn’t make a lot of sense as revision objective.
- `remove_mask`: remove the un-masking objective
- `remove_neg`: remove the negative samples from the contrastive objective
- `contrast_input_strategy`: this is what is described in the paper. Pick the most divergent hallucination as the input for the
positive example.
- `from_perturb_checkpoint`: we turn this on for the paper as it outperforms (see ablation results). The exact perturber
checkpoint is specified in the constant at the top of main.py

and will need to be updated with HuggingFace reference.

# Generating with the Reviser

## generate

Relevant hyper-parameters are:

```angular2html
parser.add_argument('--wandb_name', default='yay')
parser.add_argument('--experiment', default=None)
parser.add_argument('--gpu_device', default=None, type=int)
parser.add_argument('-only_eval', default=False, action='store_true')
```

- `only_eval`: only for the held-out eval set. This is much much faster as it processes only 1,000 sentences.
- `wandb_name`: --experiment flag passed to `main.py` training script.

# Intrinsic Evaluation of Reviser

`bash run_eval.sh {experiment}`

Data will be saved with the pattern: `{input_dir}/bhc/revise/{output,annotated,acm_output,ent_merges}/{experiment}/{example_
id}*`

To inspect raw outputs that have been decorated with hallucinations, use `_annotated` files.

# Building Revised References

## build_revised_summaries

```
parser.add_argument('--revise_experiment', default='yay')
parser.add_argument('--replace_strategy', default='balanced', choices=['balanced', 'max_coverage', 'extractive'])
```

- `revise_experiment`: `--wandb_name` or `--experiment` name from `generate.py` 
- `replace_strategy`: `max_coverage` is _Less Abstractive_ in paper, balanced is _More Abstractive_, and extractive is _Fully Extractive_

This saves the dataset into

`{input_dir}/bhc/summary_dataset_revised_{max_coverage,balanced,extractive}.csv`

and sent-level statistics into

`{input_dir}/bhc/stats_for_summary_dataset_revised_{max_coverage,balanced,extractive}`

**NB**: after generating the revised dataset, you will need to run:

```
preprocess#cache_sent_level_rouge.py --version {revised_balanced,revised_extractive,revised_max_coverage} --mode compute
```

To save the sentence-level scores for fine-tuning with Oracle extractive filtering. Followed by the same command with `--mode annotate` to include ROUGE scores for each sentence in the string (easier  than loading in the separate file during training). The output of the `cache_sent_level_rouge` will be the training data loaded in by `gen_transformers#main`.
