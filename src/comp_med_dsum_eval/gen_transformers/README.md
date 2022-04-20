# Training Models

**Script: main.py**

Feel free to change the Weights & Biases in `main.py` settings to point to your own personal **project** and **entity**.  Otherwise, they will be available at [Our Public Project](https://wandb.ai/griffinadams/mimic-sum).

```angular2html
logger = pl_loggers.WandbLogger(
    name=args.experiment,
    save_dir=experiment_dir,
    offline=args.debug or args.offline,
    project='mimic-sum',
    entity='griffinadams',
)
```

## Important Hyper-Parameters

- To train on examples with an admission note: set `-
require_admission_note`
- For faithful data only (high quality), set --
quality_weights 0,0,1 which will assign no weight to low,mid coverage examples and normal weight 1 to high
quality. This will actually filter out the training data when it sees that no weight has been assigned.
- For revised versions of the dataset, set `--version revised_{balanced,max_coverage,extractive}` and make sure that `--reviser_experiment`` points to the same `--experiment` used during training.  If there are issues, feel free to modify `get_path_from_exp` to point directly to the `.ckpt` file you want.
- To run with loss truncation, make sure to set `--dropc` to a value `> 0` and tune `drop_warmup_steps`.
- To ignore hallucinated entities during training, turn on `-ignore_hallucinated_ents`
- To control for hallucinations during training and then generate from the lowest hallucination tranche, make sure to turn on `-control_hallucinations`
- Set your huggingface weights with `--hf_model`.  For the Longformer, we used `allenai/led-base-16384` and BART `facebook/bart-base`.  Any models will work, just make sure to adjust `--max_input_length` as needed based on the desired input length.  If using a different HuggingFace model, make sure to add a new block for it in the following if chain

```angular2html
if bart_model is not None:
    self.model = bart_model
elif 'led' in hf_model:
    kwargs = {'gradient_checkpointing': True}
    self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model, **kwargs)
elif 'bart' in hf_model or 'perturber' in hf_model:
    self.model = BartForConditionalGeneration.from_pretrained(hf_model)
elif 'your-model' in hf_model:
    self.model = {MyHuggingModel}.from_pretrained(...)
else:
    assert 'pegasus' in hf_model
    self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
        hf_model, block_size=16, num_random_blocks=2)
```

# Generating Summaries

- As with the other scripts for `perturber` and `reviser`, the wandb_name is the weights directory and is set to be the experiment results / outputs path unless experiment is overidden. 
- The tl;dr - just call generate.py with `--wandb_name` unless you don’t want the outputs stored under `{args.input_dir}/{args.target}/mimic_sum/results/{args.wandb_name}` and would rather under a different `{args.input_dir}/{args.target}/mimic_sum/results/{args.experiment}`.

## Evaluating Summaries

```
bash run_ent_eval.sh {experiment}
python results_to_table.py --experiment {experiment}
```

This experiment is determined by the value of `--wandb_name` for the generate call, or overridden by setting an explicit `--experiment` flag.

Results-to-table will aggregate the results to match what is shown in the main results Table (4) of the paper.

To run the entailment model, first run `retrieve_contexts.py` and then run `../eval/entailment.py` which will treat the aligned source from `retrieve_contexts.py` as the premise and the generated sentence as the hypothesis.  Scoring is done from the SciFive [model](https://huggingface.co/razent/SciFive-large-Pubmed_PMC) fine-tuned on [MedNLI](https://physionet.org/content/mednli/1.0.0/).