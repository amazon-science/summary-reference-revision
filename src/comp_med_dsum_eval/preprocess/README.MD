
# Setup

Request access and download the MIMIC-III database from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) and specify save location in scripts with `--input_dir`. 

For the paper, we extract the Brief Hospital Course section of the discharge summary. In the code,
this is done by calling scripts with `--target bhc`
All data is saved under `{input_dir}/bhc` which is the concatenation of two flags: `--input_dir` and `--target`

# Generating MIMIC-III summarization corpus

1. `get_relevant_notes.py --target bhc`
2. `structure_notes --dtypes source,target`
3. `collect_dataset.py`
4. `extract_entities --dtypes source,target`
5. `compute_idf.py`
6. `gen_splits.py`
7. `cache_sent_level_rouge.py`
8. `cache_hf_outputs.py`

These scripts run with multiple processes for speed which can be modulated with the `--cpu_frac` argument.

## get_relevant_notes

- Filter out MIMIC-III notes for those pertaining to a visit with a valid discharge summary (includes the target section of
interest).

## structure_notes

Parses each note into an html like structure to demarcate sections, sub-sections, sentences, and notes. Later on, we add
entity information with its own tag. All tags are closed, i.e., `</d>`.


```angular2html
<d> = note
<h> = section
<p> = sub-section
<s> = sentence
<e> = entity
```

Every HTML tag and body is delimited by `<SEP>`

Utilities in `preprocess/sec_tag/section_utils` help with parsing this html. Most commonly used are shown below. Feel free to
write your own and debug on `{input_dir}/bhc/summary_dataset_mini.csv`. `HTML_REGEX_NO_SPACE` in
`preprocess/constants.py` is a regex that finds open and close HTML tags and is useful for splitting, identification etc. It does
not look for uppercase tags so you must either split by `<SEP>` or call `html_str.replace(‘<SEP>’, ‘’)` separately.

```
def remove_tags_from_sent - returns raw note text without any HTML structure.
def sents_from_html - returns a list of sentences from an HTML note (no section headers).
```

You can pass a whole note as well as a single sentence (with entities decorated with HTML).

## collect_dataset

This creates a file /efs/griadams/bhc/summary_dataset.csv that includes an example id “SUBJECT_ID” + “_” +
“HADM_ID” a source column, and a target column. The source is just the concatenation of all the source notes for the example: `<d note_id=x>...</d><d note_id=y>...</d>`.

Note ids are constructed from the ROW_ID, HADM_ID, and SUBJECT_ID.

```angular2html
def add_note_id(notes_df):
   note_id_cols = ['ROW_ID', 'HADM_ID', 'SUBJECT_ID']
   print('Compiling note_id column...')
   notes_df['note_id'] = notes_df[note_id_cols].apply(
   lambda row: 'mimic-' + '-'.join([str(x).split('.')[0] for x in row.values.asty
```

## compute_idf

Saves tokens and corpus counts (with IDF) to `{input_dir}/bhc/toks{_uncased}.csv`
(-uncased is a flag).

## gen_splits


This generates the train-test-splits and saves each data split to

```
{input_dir}/bhc/summary_dataset_train.csv
{input_dir}/bhc/summary_dataset_validation.csv
{input_dir}/bhc/summary_dataset_test.csv
```

We also save just the example ids to

```
{input_dir}/bhc/summary_dataset_train_example_ids.csv
{input_dir}/bhc/summary_dataset_validation_example_ids.csv
{input_dir}/bhc/summary_dataset_test_example_ids.csv
```

These splits should be the same as used in the paper given the same random seed and `NOTEEVENTS.csv` file from MIMIC-III.

**IMPORTANT**: You can get the training set in one of 2 ways
1. Load `{input_dir}bhc/summary_dataset_train.csv`
2. Load `{input_dir}/bhc/summary_dataset.csv` and filter for example ids
in `{input_dir}/bhc/summary_dataset_train_example_ids.csv`


(I generally use the latter — see `gen_transformers.dataset.add_meta`)

## cache_sent_level_rouge

For each sentence in the source, we compute its ROUGE-1, ROUGE-2, and ROUGE-L F1-score against the reference. --
mode compute computes the scores and saves them to disc.

The resulting ROUGE scores and ranks are packed into the sentence-level tags:

```<s idx={sent_idx} score={score} rank={rank}> This is a source sentence </s>```

This is used for oracle sentence filtering for `BART` and `Longformer` as needed.  Feel free to bypass this filtering step and implement your own two step extract-then-abstract pipeline.

## cache_hf_outputs

This computes `clinicalBERT` and `bioELECTRA` embeddings for all the sentences in `summary_dataset.csv`. Thus for
original references and source sentences, we don’t need to recompute when aligning data or performing evaluations for
perturber, reviser, or summarization. This will take up a lot of memory so only do if you plan on recreating the evaluations in the paper.

# Entity Extraction (`./entity`)

Script order

1. `extract_ents.py`
2. `process_ents.py`
3. `ent_types.py`

## extract_ents

This extracts entities for all text in `summary_dataset.csv` from `icd`, `rxnorm`, and `entities_v2`.

This requires access to an Amazon Comprehend Medical account.  If you do not have this, please follow the target output format with your own entity extraction toolkit, i.e., CLAMP, scispacy, cTAKES, MedCAT, Stanza.

saves a json file of entities for each example under `{input_dir}/bhc/acm_output/{example_id}.json`

Each JSON output has two keys: one for `source_ents` and one for `target_ents`.

## process_ents

Process ents computes a merge score for every source-target entity pair to determine if they are synonyms.

See below code snippet for instruction on how to load an entity file, filter entities (remove overlapping entities, duplicates, and entity types we don’t care about
— like demographics), add entity ids (unique identifier), and flatten the entity object to be a list of entity objects.
where fn is `{input_dir}/bhc/acm_output/{example_id}.json`

```angular2html
import itertools
import ujson

with open(fn, 'r') as fd:
    ents = ujson.load(fd)
source_ents = ents['source']
target_ents = ents['target']
source_ents_flat = list(itertools.chain(
   *[filter_ents(add_ent_id(ent_obj, 'source')) for ent_obj in source_ents]
))
target_ents_flat = list(itertools.chain(
   *[filter_ents(add_ent_id(ent_obj, 'target')) for ent_obj in target_ents]
))
```

The output of `process_ents` are files of the format `{input_dir}/bhc/acm_output/{example_id}.csv`.

Importantly, columns `should_merge` (boolean) indicates whether or not to merge `source_ent_id`
and `target_ent_id`. Merge for us means that `target_ent_id=x` is NOT a hallucination iff there exists >0 rows in the
merge.csv where `target_ent_id=x` AND `should_merge`.

## ent_types

Records an inventory of entities for all extracted entities in `{input_dir}/bhc/acm_output/{example_id}.json`.  Maps to the following types:

```angular2html
ENT_ORDER = ['dx', 'procedure', 'treatment', 'test', 'med']
ENT_TYPE_MAP = {
   'DX_NAME': 'dx',
   'PROCEDURE_NAME': 'procedure',
   'TREATMENT_NAME': 'treatment',
   'TEST_NAME': 'test',
   'BRAND_NAME': 'med',
   'GENERIC_NAME': 'med'
}
```

This just merges brand and generic drugs to the category “med”. This ent inventory is used for the random entity swap
baseline when randomly sampling entities to swap by type.  Please over-write these maps with your own semantic groupings (i.e., UMLS semantic groups).

