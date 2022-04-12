#!/bin/sh
EXPERIMENT="$1"
echo $EXPERIMENT
#sudo python extract_ents_for_revised.py --experiment $EXPERIMENT
#sudo python process_ents_for_revised.py --experiment $EXPERIMENT
#sudo python ent_evaluation.py --experiment $EXPERIMENT
sudo python evaluate.py --experiment $EXPERIMENT
#sudo python agg_eval.py --experiment $EXPERIMENT
