# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import load_metric
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import spacy
import torch
from transformers import (
    AutoModelForSeq2SeqLM, BigBirdPegasusForConditionalGeneration, BartForConditionalGeneration, AutoTokenizer,
    AutoModelForPreTraining, AutoModel, BertForNextSentencePrediction
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother

from comp_med_dsum_eval.gen_transformers.loss_dropper import LossDropper
from comp_med_dsum_eval.preprocess.sent_segment.main import parse_sentences_spacy
from comp_med_dsum_eval.perturber.evaluate import encode, plausibility, bertscore_post_filt
from comp_med_dsum_eval.ref_reviser.build_train_dataset_from_perturbed import delete, stopword_idxs


def nsp(s1, s2, nsp_model, nsp_tokenizer):
    dummy_label = torch.LongTensor([1]).to(nsp_model.device)
    encoding = nsp_tokenizer(s1, s2, return_tensors='pt', max_length=512, truncation=True)
    encoding = {k: v.to(nsp_model.device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = nsp_model(**encoding, labels=dummy_label)
    logits = outputs.logits
    prob = torch.softmax(logits, dim=1)
    return float(prob[0, 0])


def h_no_special(tokens, h, seq_len):
    remove_idxs = stopword_idxs(tokens)
    h_no_stop_no_pad = h[:seq_len][1:-1]
    if len(remove_idxs) < len(h_no_stop_no_pad):
        h_no_stop_no_pad = delete(h_no_stop_no_pad, remove_idxs)
    return h_no_stop_no_pad


def avg_nsp(sents, nsp_tokenizer, nsp_model):
    n = len(sents)
    if n <= 1:
        return None
    nsps = []
    for i in range(n - 1):
        nsps.append(nsp(sents[i], sents[i + 1], nsp_model, nsp_tokenizer))
    return sum(nsps) / float(len(nsps))


def get_eval_models(device='cuda'):
    electra_tokenizer = AutoTokenizer.from_pretrained('sultan/BioM-ELECTRA-Large-Discriminator')
    electra_model = AutoModelForPreTraining.from_pretrained('sultan/BioM-ELECTRA-Large-Discriminator')

    clinbert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    clinbert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    nsp_model = BertForNextSentencePrediction.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    return {
        'electra': {'model': electra_model.to(device), 'tokenizer': electra_tokenizer},
        'bert': {'model': clinbert_model.to(device), 'tokenizer': clinbert_tokenizer},
        'nsp': {'model': nsp_model.to(device), 'tokenizer': clinbert_tokenizer}
    }


class TransformerSummarizer(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_model, bart_model=None):
        """
        bart_model -> can load in pre-trained bart weights outside of this function (from reviser checkpoint)
        """
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        if bart_model is not None:
            self.model = bart_model
        elif 'led' in hf_model:
            kwargs = {'gradient_checkpointing': True}
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model, **kwargs)
        elif 'bart' in hf_model or 'perturber' in hf_model:
            self.model = BartForConditionalGeneration.from_pretrained(hf_model)
        else:
            assert 'pegasus' in hf_model
            self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
                hf_model, block_size=16, num_random_blocks=2)
        # If we restore from reviser checkpoint or perturber checkpoint there will be extra tokens not in bart-base
        # see perturber.main and ref_reviser.main for additional tokens
        self.model.resize_token_embeddings(len(tokenizer))
        self.lr = self.hparams.lr
        self.train_size = None
        self.rouge = load_metric('rouge')
        self.label_smoother = LabelSmoother(epsilon=0.1)
        effective_warmup = self.hparams.drop_warmup_steps * self.hparams.grad_accum
        self.dropper = None if self.hparams.dropc == 0 else LossDropper(
            dropc=self.hparams.dropc, min_count=effective_warmup, recompute=effective_warmup
        )
        self.eval_models = None

        print('Loading Spacy...')
        self.sentencizer = spacy.load('en_core_sci_sm', disable=['ner', 'parser', 'lemmatizer'])
        self.sentencizer.add_pipe('sentencizer')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Reviser Finetuning')
        parser.add_argument('--lr', type=float, default=2.2e-4)
        parser.add_argument('--target_batch_size', type=int, default=16)
        parser.add_argument('--weight_decay', type=float, default=5e-5)
        parser.add_argument('--max_output_length', type=int, default=1024)
        parser.add_argument('--max_input_length', type=int, default=16384)
        parser.add_argument('--quality_weights', type=str, default='1,1,1',
                            help='LM-objective weights based on quality metric bucket: low,mid,high')
        parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
            'allenai/led-base-16384',
            '/efs/griadams/dsum/weights/perturber/ent_sample/checkpoint-100000',
            '/efs/griadams/dsum/weights/perturber/no_ent/checkpoint-100000',
            'facebook/bart-base',
        ])
        return parent_parser

    def on_validation_start(self):
        self.eval_models = get_eval_models(self.device)

    def on_predict_start(self):
        self.eval_models = get_eval_models(self.device)

    def on_validation_end(self):
        self.eval_models = None

    def on_predict_end(self):
        self.eval_models = None

    def training_step(self, batch, batch_idx):
        quality = batch.pop('quality')[0]
        has_admission_note = batch.pop('has_admission_note')[0]
        ignore_idxs = batch.pop('ignore_idxs', [])
        if quality is None:
            weight = 1.0
        elif quality == 'low':
            weight = self.hparams.low_weight
        elif quality == 'mid':
            weight = self.hparams.mid_weight
        elif quality == 'high':
            weight = self.hparams.high_weight
        else:
            raise Exception(f'Unrecognized quality bucket -> {quality}')
        output = self.model(**batch, use_cache=False)
        loss = output.loss
        self.log('train_loss', loss, on_epoch=False, on_step=True, prog_bar=True)
        if quality is not None:
            self.log(f'train_{quality}_loss', loss, on_epoch=False, on_step=True, prog_bar=True)
        should_keep = self.dropper is None or self.dropper.should_keep(loss)
        if not should_keep:
            weight = 0  # Skip it

        loss_labels = batch['labels'].clone()
        # Are we ignoring hallucinated entities when calculating smoothed log loss?
        for batch_idx, ignore_idx in ignore_idxs:
            loss_labels[batch_idx, ignore_idx] = -100
        smooth_loss = self.label_smoother(output, batch['labels'])
        return weight * smooth_loss

    def rouge_metrics(self, generated, gold):
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        rouge_output = self.rouge.compute(predictions=generated, references=gold, rouge_types=rouge_types)

        stats = {}
        f1s = []
        for rouge_type in rouge_types:
            stats[f'{rouge_type}_precision'] = rouge_output[rouge_type].mid.precision
            stats[f'{rouge_type}_recall'] = rouge_output[rouge_type].mid.recall
            stats[f'{rouge_type}_f1'] = rouge_output[rouge_type].mid.fmeasure
            f1s.append(rouge_output[rouge_type].mid.fmeasure)
        stats['mean_f1'] = np.array(f1s).mean()
        return stats

    def shared_generate(self, batch, num_beams=1):
        kwargs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'use_cache': True,
            'num_beams': num_beams,
            'min_length': 128,
            'max_length': self.hparams.max_output_length,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
        }

        if 'global_attention_mask' in batch:
            kwargs['global_attention_mask'] = batch['global_attention_mask']
        generated_ids = self.model.generate(**kwargs)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        output_ids = batch['labels']
        output_ids[torch.where(batch['labels'] == -100)] = 1
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        return generated_str, gold_str

    def validation_step(self, batch, batch_idx):
        quality = batch.pop('quality')[0]
        has_admission_note = batch.pop('has_admission_note')[0]
        output = self.model(**batch)
        loss = output.loss

        generated_str, gold_str = self.shared_generate(batch)
        metrics = self.rouge_metrics(generated_str, gold_str)
        overlap_metrics = self.overlap_metrics(generated_str[0], gold_str[0])
        metrics.update(overlap_metrics)
        for k, v in metrics.items():
            if v is None:
                continue
            self.log(k, v, on_epoch=True, on_step=False, prog_bar=True)
            if quality is not None:
                self.log(f'{k}_{quality}', v, on_epoch=True, on_step=False, prog_bar=True)
            if has_admission_note:
                self.log(f'{k}_admission', v, on_epoch=True, on_step=False, prog_bar=True)

        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        nps = list(self.named_parameters())
        grouped_parameters = [
            {
                'params': [p for n, p in nps if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in nps if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr)
        if self.hparams.no_schedule or self.hparams.debug or self.hparams.find_lr:
            return optimizer

        # 6% is somewhat standard for fine-tuning Transformers (can be a tunable hyper-parameter as well)
        # nonzero warmup helps mitigate risk of catastrophic forgetting from pre-training (big risk bc/ of new domain)
        # warmup = round(0.06 * self.hparams.max_steps)
        warmup = 200
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=self.hparams.max_steps)

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]

    def overlap_metrics(
            self, generated_str, reference_str, bert_source_h=None, electra_source_h=None, include_ref=False):
        metrics = {}
        pred_sents = parse_sentences_spacy(generated_str, self.sentencizer)
        ref_sents = parse_sentences_spacy(reference_str, self.sentencizer)
        all_sents = pred_sents + ref_sents
        electra_tokens = list(map(self.eval_models['electra']['tokenizer'].tokenize, all_sents))
        electra_outputs, electra_seq_lens = encode(
            all_sents, self.eval_models['electra']['model'], self.eval_models['electra']['tokenizer'],
            device=self.device, max_length=256, top_n_layers=4)
        electra_h_clean = []
        for i in range(len(all_sents)):
            electra_h_clean.append(h_no_special(
                electra_tokens[i], electra_outputs['hidden_states'][i], electra_seq_lens[i]
            ))

        pred_electra_h = torch.cat(electra_h_clean[:len(pred_sents)], dim=0)
        ref_electra_h = torch.cat(electra_h_clean[len(pred_sents):], dim=0)
        electra_prec, electra_cov, electra_f1 = bertscore_post_filt(pred_electra_h, ref_electra_h)
        if electra_source_h is not None:
            electra_src_prec, electra_src_cov, electra_src_f1 = bertscore_post_filt(pred_electra_h, electra_source_h)
            metrics.update({
                'electra_bs_src_cov': electra_src_cov,
                'electra_bs_src_prec': electra_src_prec,
                'electra_bs_src_f1': electra_src_f1,
            })

        metrics.update({
            'electra_bs_cov': electra_cov,
            'electra_bs_prec': electra_prec,
            'electra_bs_f1': electra_f1,
        })

        logits = []
        for i in range(len(electra_tokens)):
            raw_logits = electra_outputs['logits'][i, :electra_seq_lens[i]]
            logits.append(raw_logits[1:-1].float())
        fake_metrics = [plausibility(electra_tokens[i], logits[i]) for i in range(len(electra_tokens))]

        fake_metrics_pred = pd.DataFrame(fake_metrics[:len(pred_sents)])
        fake_metrics_ref = pd.DataFrame(fake_metrics[len(pred_sents):])
        for k in list(fake_metrics_pred.columns):
            metrics['ref_' + k] = fake_metrics_ref[k].mean()
            metrics['pred_' + k] = fake_metrics_pred[k].mean()

        bert_tokens = list(map(self.eval_models['bert']['tokenizer'].tokenize, all_sents))
        bert_outputs, bert_seq_lens = encode(
            all_sents, self.eval_models['bert']['model'], self.eval_models['bert']['tokenizer'],
            self.device, max_length=128, top_n_layers=4
        )
        bert_h_clean = []
        for i in range(len(all_sents)):
            bert_h_clean.append(h_no_special(
                bert_tokens[i], bert_outputs['hidden_states'][i], bert_seq_lens[i]
            ))

        pred_bert_h = torch.cat(bert_h_clean[:len(pred_sents)], dim=0)
        ref_bert_h = torch.cat(bert_h_clean[len(pred_sents):], dim=0)
        bert_prec, bert_cov, bert_f1 = bertscore_post_filt(pred_bert_h, ref_bert_h)

        if bert_source_h is not None:
            bert_src_prec, bert_src_cov, bert_src_f1 = bertscore_post_filt(pred_bert_h, bert_source_h)
            metrics.update({
                'bert_bs_src_cov': bert_src_cov,
                'bert_bs_src_prec': bert_src_prec,
                'bert_bs_src_f1': bert_src_f1,
            })

        metrics.update({
            'bert_bs_cov': bert_cov,
            'bert_bs_prec': bert_prec,
            'bert_bs_f1': bert_f1,
        })

        metrics['pred_nsp'] = avg_nsp(
            pred_sents, self.eval_models['nsp']['tokenizer'], self.eval_models['nsp']['model'])
        if include_ref:
            metrics['ref_nsp'] = avg_nsp(
                ref_sents, self.eval_models['nsp']['tokenizer'], self.eval_models['nsp']['model'])
        return metrics

    def predict_step(self, batch, batch_idx=None, bert_source_h=None, electra_source_h=None):
        example_id = batch.pop('example_id')[0]
        quality = batch.pop('quality')[0]
        has_admission_note = batch.pop('has_admission_note')[0]
        generated_str, gold_str = self.shared_generate(batch, num_beams=4)
        outputs = {'example_id': example_id}
        outputs.update(self.rouge_metrics(generated_str, gold_str))
        outputs.update(self.overlap_metrics(
            generated_str[0], gold_str[0], bert_source_h=bert_source_h, electra_source_h=electra_source_h))
        source = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=True)[0]
        outputs['source'] = source
        outputs['quality'] = quality
        outputs['has_admission_note'] = has_admission_note
        outputs['prediction'] = generated_str[0]
        outputs['target'] = gold_str[0]
        return outputs

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items
