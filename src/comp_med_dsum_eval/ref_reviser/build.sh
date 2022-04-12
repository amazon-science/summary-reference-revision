#!/bin/sh
EXPERIMENT="$1"
STRATEGY="$2"
echo $EXPERIMENT
echo $STRATEGY
sudo python generate.py --wandb_name $EXPERIMENT
sudo ./run_eval.sh $EXPERIMENT
sudo python build_revised_summaries.py --revise_experiment $EXPERIMENT --replace_strategy $STRATEGY
sudo python ../preprocess/cache_sent_level_rouge.py --version "revised_${STRATEGY}" --reviser_experiment $EXPERIMENT
