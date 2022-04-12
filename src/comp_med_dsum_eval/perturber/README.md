- Training Data comes from `./dsum_data/` (**d**ischarge **sum**maries). 
- The data for generation comes from `./sum_data/` (unsupported reference sentences from the **sum**marization corpus)

# ./dsum_data

- dsum data for training the perturber is all stored under `{input_dir}/dsum` (it’s not section specific).

**Script Order**
1. `structure_notes`
2. `header_concepts`
3. `extract_ents`
4. `generate_sent_idx`
5. `retrieve_related_ents --mode query (followed by --mode process`

## structure_notes

- Collects all discharge summaries, structures them into the html format described above and outputs them to
  `{input_dir}/dsum/dsum_tagged`.

## header_concepts

- Removes sentences from irrelevant section header concepts as specified in the file. This filters out irrelevant sections and
stores transforms `{input_dir}/dsum/dsum_tagged.csv` into `{input_dir}/dsum/dsum_tagged_filt.csv`.

## extract_ents

- This extracts entities for the discharge summaries in the same way we extract for each source-target text in
`preprocess.entity#extract_ents`.

## generate_sent_idx

Generates a Faiss sentence index with BioSentVec for each sentence in the discharge summary and annotates entity
information (span and entity type) in the HTML. This is necessary because when we retrieve sentences for related entity
search, we can easily extract the entities out of them just from the string.


This saves 3 files under `{input_dir}/dsum/`:

```angular2html
sent_index.bin - Faiss index
sent_embeds.pk - BioSentVec embeddings for each sentence (used for querying)
sent_index_meta.csv - entity-demarcated sentences for each of the dsum sentences in the index
```

## retrieve_related_ents

For each sentence in sent_index_meta, it retrieves its closest 250 neighbors from sent_index.bin (Faiss index) and adds
columns related to each entity type (med, dx, treatment, procedure, test). maximum of 25 per each type. You can tune hyper-parameters here:

```angular2html
parser.add_argument('--top_k_sents', default=250, type=int)
parser.add_argument('--top_k_ents', default=25, type=int)
```

Changing this will change the topicality and divergence of hallucinations.

- `top_k_sents` refers to how many sentences to retrieve from the index and `top_k_ents` how many entities of each type to add
to the distractor set before stopping. Higher values of both will lead to entities chosen that are topically further away. 

This script saves the entity information, along with the sentence info from `sent_index_meta.csv` to `sent_index_meta_w_related_ents.csv`.  Then, `sent_index_meta_w_related_ents.csv` is the training data for the perturber.

# Training the Perturber

Training is done in `main.py`.

The main tunable hyperparameters are:

```
parser.add_argument('-no_ent', default=False, action='store_true')
parser.add_argument('--seed', default=1956, type=int)
parser.add_argument('--max_steps', default=100000, type=int)
```

`no_ent` is for the paper ablation (No Ent Swap). It just removes the entity distractor sets and doesn’t perform entity
swapping. All the corruptions described in the paper happen in the `perturber/dataset.py` file.

IMPORTANT: the `--experiment` flag will specify the `wandb_name` and name of the run so choose it and remember it.

# Generating with the Perturber

## generate_perturbations

```
parser = argparse.ArgumentParser('Generating Perturbations of High Quality Referen
parser.add_argument('--input_dir', default='/efs/griadams')
parser.add_argument('--wandb_name', default=None, required=True)
parser.add_argument('--experiment', default=None)
parser.add_argument('--target', default='bhc', choices=['hpi', 'bhc'])
parser.add_argument('-debug', default=False, action='store_true')
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('-only_new', default=False, action='store_true')
parser.add_argument('-no_ent', default=False, action='store_true')
parser.add_argument('-only_eval', default=False, action='store_true')
parser.add_argument('--max_n', default=99999999999, type=int)
parser.add_argument('--seed', default=1992, type=int)
parser.add_argument('--sample_strategy', default='resample', choices=['resample',
parser.add_argument('--chunksize', default=8, type=int)
parser.add_argument('--chunk_idx', default=None, type=int)
parser.add_argument('--gpu_device', default=None, type=int)
# Ablations (for paper)
parser.add_argument('-no_ent_trick', default=False, action='store_true', help='Remove entity trick
parser.add_argument('--ent_ctrl_add', default=1, type=int)
```

**Import parameters to set**

`--wandb_name`: this is the name of the run as it is saved on wandb (which is specified by --experiment when training in
main.py

`-only_eval`: only generate for the evaluation subset. Do this for intrinsic evaluation models only, not if you want to use the
outputs for the revision model.

`-no_ent`: make sure this is turned on if you are evaluating the model for a model trained without the entity swaps

`--chunk_idx`: run the script on the chunk_idx / chunksize ‘th chunk of the generation data. This if you want to run multiple
scripts on different chunks and different gpu_device.

**Inference Tricks**

`-no_ent_trick`: turning this on removes the entity removing trick during inference. You should only set this flag for the
ablation.

`--ent_ctrl_add`: how much to add to the ent-add and ent-remove codes during inference.  This is the *add-1* trick described in the paper.

## Creating a pure entity swap (non-parametric baseline)

This baseline can be found in `generate_swaps.py`.

```angular2html
parser.add_argument('--strategy', default='random', choices=['random', 'related'])
```

`random` samples an entity of the same type from the entity inventory (created during  preprocessing), and `related` from an entity from the distractor set.

# Intrinsically Evaluating Perturber

**Script Order**
1. `bash run_ent_evaluation.sh {experiment_name}`
2. `results_to_table.py —experiment {experiment name}`

## run_ent_evaluation

Runs all the evaluation scripts back to back and then runs `agg_eval` which will aggregate the results into a `results.csv` file. `results_to_table` just prints this in the same order as it is rendered in the evaluation table for easy
copy and pasting into excel or overleaf.

The outputs of the evaluations and generations will be in directory (and it will be one file per example for each sub-directory)
`{input_dir}/perturb/{experiment}/*`
- `output/{example_id}.csv` — raw outputs (perturb_idx from 0-4 for 5 different samples. We drop exact match duplicate generated strings)
- `nsp/{example_id}.csv` - average nsp scores
- `annotated/{example_id}.csv` — entity decorated outputs (useful for figures and marking entities as hallucinated or not). global halluc is 1 if the entity is novel to the entire source text and local if it’s just novel to the input sentence.
- `acm_output/{example_id}.json` - output of the entity extraction step for the perturber
- `ent_merges/{example_id}.csv` - output for the entity merging (from acm_ouput). What gets used to determine faithfulness.
- `ent_eval/{example_id}.csv` - entity evaluation metrics for each example_id, target sent_idx, perturb_idx output
- `eval/{example_id}.csv` - BERTScore overlaps and ELECTRA outputs

## results_to_table

Pass the experiment in and it will spit out all the results for the paper table on intrinsic evaluation of the perturber model in the Appendix.
