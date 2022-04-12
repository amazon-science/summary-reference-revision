# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer, BartForConditionalGeneration

import numpy as np
from comp_med_dsum_eval.gpu_utils import get_free_gpus
from random import gauss, shuffle


PATH = '/Users/griffinadams/Desktop/ref_sum/perturber_checkpoint'

def ent_str(ents, tag):
    return f'<{tag}> ' + ' <ent-sep> '.join(ents) + f' </{tag}>'


def get_model(path, device):
    print(f'Loading model and tokenizer from {path}')
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = BartForConditionalGeneration.from_pretrained(path).to(device)
    return model, tokenizer


def generate(m, t, text, device):
    model_inputs = t(text, add_special_tokens=True, padding='longest', truncation=True, max_length=1024, return_tensors='pt')

    kwargs = {
        'input_ids': model_inputs['input_ids'].to(device),
        'attention_mask': model_inputs['attention_mask'].to(device),
        'use_cache': True,
        'num_beams': 4,
        'min_length': 5,
        'max_length': 128,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
    }
    generated_ids = m.generate(**kwargs)
    generated_strs = t.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
    return generated_strs

# example_id 55672_193172
# sent_idx 81

prefix = '<ent-remove-4> <ent-add-2> <span-remove-11> <shuffle-5>'
med = 'Klonopin<SEP>Dilaudid<SEP>Haldol<SEP>benadryl<SEP>sertraline<SEP>Ativan<SEP>lithium<SEP>Zyprexa<SEP>tylenol<SEP>Benzodiazepines<SEP>Trazodone<SEP>Prozac<SEP>Risperdal<SEP>Clonidine<SEP>methadone<SEP>antibiotics<SEP>levothyroxine<SEP>Coumadin<SEP>levofloxacin<SEP>levetiracetam<SEP>stool softeners<SEP>lactulose<SEP>Soma<SEP>steroids<SEP>morphine'
dx = 'polysubstance abuse<SEP>asthma<SEP>seizure disorder<SEP>head injury<SEP>depression<SEP>anxiety<SEP>rhabdomyolysis<SEP>transaminitis<SEP>change in mental status<SEP>airway protection<SEP>hypercarbic respiratory failure<SEP>alcohol abuse<SEP>hypertension<SEP>hypothyroidism<SEP>bipolar disorder<SEP>delirium tremens<SEP>pneumonia<SEP>withdrawal seizures<SEP>homelessness<SEP>Hep B C<SEP>tremulousness<SEP>tachy<SEP>hypertensive<SEP>suicide attempts<SEP>hypoxia'
procedure = 'intubated<SEP>extubation<SEP>sedation<SEP>sedated<SEP>banding<SEP>tracheostomy<SEP>trach inserted<SEP>surgery<SEP>intubation<SEP>stenting<SEP>spinal surgery<SEP>Port-A-Cath<SEP>evacuation<SEP>ablation<SEP>dental extractions<SEP>knee surgery<SEP>chest tube placement<SEP>temporal lobectomy<SEP>liver transplant<SEP>ORIF<SEP>choleycystectomy<SEP>open reduction internal fixation<SEP>repaired<SEP>en-Y gastric bypass surgery<SEP>multiple abdominal surgeries'
treatment = 'Intensive Care<SEP>IV narcotics<SEP>high-dose narcotics<SEP>sleeping pills<SEP>chronic narcotics<SEP>IV antibiotics<SEP>psychiatric medication<SEP>narcotics<SEP>sedating meds<SEP>Brachytherapy<SEP>pressors<SEP>psychiatric meds<SEP>vasopressors<SEP>prolonged intubation<SEP>paracenteses<SEP>thoracenteses<SEP>IV medications<SEP>DBT treatment<SEP>packed red blood cells<SEP>hydration<SEP>outpatient psychiatric medications<SEP>sedating medications<SEP>ECT treatments<SEP>hemodialysis<SEP>steroid'
test = 'temp<SEP>ETOH level<SEP>sat<SEP>CIWA scale<SEP>blood draws<SEP>imaging exams<SEP>lab<SEP>LVEF<SEP>stress test<SEP>UA<SEP>BAL<SEP>random cortisol<SEP>Na<SEP>TSH<SEP>SBP<SEP>CT head<SEP>neurorads<SEP>physical signs<SEP>drawel CIWA scale<SEP>brain biopsy<SEP>hematocrit<SEP>WBCs<SEP>Tbili<SEP>WBC<SEP>bands'

original = 'Several days into hospital course, patient developed delirium and agitation most likely secondary to long hospital course and multiple psychotropic medications as well as history of EtOH abuse.'
removed_ents = 'Several days into hospital course, patient developed and most likely secondary to long hospital course and multiple as well as history of EtOH abuse.'
removed_span = 'and most likely secondary to long hospital course and multiple as'
frag = 'Several days into hospital course, patient developed and well as history of EtOH abuse.'
entities_to_insert = ['Risperdal']

def insert_ents_randomly(toks, ents_to_add):
    for ent in ents_to_add:
        idx = int(np.random.randint(0, len(toks) + 1))
        toks = toks[:idx] + ent.split(' ') + toks[idx:]
    return toks


def partial_shuffle(arr, orderliness=0.5):
    # https://stackoverflow.com/questions/62436299/how-to-lightly-shuffle-a-list-in-python
    def _tuplify(x, y):
        return orderliness * y + gauss(0, 1), x

    pairs = list(map(_tuplify, arr, range(len(arr))))
    pairs.sort()
    partially_ordered_values = [p[1] for p in pairs]
    return partially_ordered_values

inputs = []
for _ in range(5):
    dx_rand = list(np.random.choice(dx.split('<SEP>'), size=(10,), replace=False))
    test_rand = list(np.random.choice(test.split('<SEP>'), size=(10,), replace=False))
    procedure_rand = list(np.random.choice(procedure.split('<SEP>'), size=(10,), replace=False))
    treatment_rand = list(np.random.choice(procedure.split('<SEP>'), size=(10,), replace=False))
    med_rand = list(np.random.choice(med.split('<SEP>'), size=(10,), replace=False))
    frag_toks = insert_ents_randomly(frag.split(' '), entities_to_insert)
    frag_corrupt = ' '.join(partial_shuffle(frag_toks, orderliness=0.5))
    input = prefix + ' <sep> ' + ent_str(dx_rand, 'dx') + ' ' + ent_str(procedure_rand, 'procedure') + ' ' + \
                ent_str(treatment_rand, 'treatment') + ' ' + ent_str(test_rand, 'test') + ' ' + ent_str(med_rand, 'med') + ' <sep> ' + frag
    inputs.append(input)

# devices = get_free_gpus()
device = 'cpu'
# device = 'cpu' if len(devices) == 0 else devices[0]
model, tokenizer = get_model(PATH, device)
print('\n'.join(generate(model, tokenizer, inputs, device)))