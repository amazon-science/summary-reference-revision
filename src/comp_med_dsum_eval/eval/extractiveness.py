import pandas as pd
from tqdm import tqdm

from comp_med_dsum_eval.ref_reviser.dataset import tokenize
from comp_med_dsum_eval.preprocess.fragment_utils import parse_extractive_fragments


if __name__ ==  '__main__':
    experiment = 'long_revised_balanced'
    df = pd.read_csv(f'/efs/griadams/bhc/results/{experiment}/outputs.csv')

    records = df.to_dict('records')
    all_frags = []
    for record in records:
        source_toks = tokenize(remove_tags_from_sent(''.join(record['source'].split('<SEP>'))))
        pred_toks = tokenize(remove_tags_from_sent(record['prediction']))
        frags = parse_extractive_fragments(source_toks, pred_toks, remove_stop=True)
        all_frags.append(frags)

    all_frags = pd.DataFrame(all_frags)
    for col in ['compression', 'coverage', 'density']:
        print(col, all_frags[col].dropna().mean())
