# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import regex as re

import pytorch_lightning as pl
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_metric
from comp_med_dsum_eval.preprocess.sec_tag.section_utils import remove_tags_from_sent

from comp_med_dsum_eval.perturber.evaluate import bertscore, encode


def remove_prefix_and_tags(text, prefix_delim='<sep>'):
    match = re.search(rf'{prefix_delim}', text)
    if match is not None:
        text = text[match.end():].lstrip()
    text = remove_tags_from_sent(text)
    text = re.sub(r'(<[a-z-]+[^>]+>|<\/?[a-z-]+>)', '', text)
    return text


def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    pad_mask = target.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    target.clamp_min_(0)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    nll_loss.masked_fill_(pad_mask, 0.)
    smooth_loss.masked_fill_(pad_mask, 0.)
    if reduce:
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = pad_mask.numel() - pad_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smooth_loss = smooth_loss.sum() / num_active_elements
    loss = (1. - epsilon) * nll_loss + (epsilon * smooth_loss / lprobs.size(-1))
    return loss, nll_loss


def label_smoothed_unlikelihood(probs, targets, reduce=True):
    probs = probs.view(-1, probs.size(-1))
    one_minus_probs = torch.clamp(1.0 - probs, min=1e-20)
    lprobs = torch.log(one_minus_probs)
    targets = targets.view(-1, 1)
    loss, nll_loss = label_smoothed_nll_loss(lprobs, targets, ignore_index=-100, reduce=reduce)
    return loss, nll_loss


class TransformerReviser(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.hf_model)
        self.bart.resize_token_embeddings(len(tokenizer))
        self.lr = self.hparams.lr
        self.rouge = load_metric('rouge')

    def on_validation_start(self):
        self.clinbert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.clinbert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.clinbert_model.eval()
        self.clinbert_model.to(self.device)

    def on_validation_end(self):
        self.clinbert_model.cpu()
        self.clinbert_model = None
        self.clinbert_tokenizer = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Reviser Finetuning')
        parser.add_argument('--target_batch_size', type=int, default=64)
        parser.add_argument('--weight_decay', type=float, default=5e-5)
        parser.add_argument('--max_output_length', type=int, default=128)
        parser.add_argument('--max_input_length', type=int, default=1024)
        parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
            'facebook/bart-base',
            'facebook/bart-large'
        ])
        parser.add_argument('--lr', default=5e-5, type=float)
        parser.add_argument('-from_perturb_checkpoint', default=False, action='store_true')
        parser.add_argument('--max_steps', default=50000, type=int)
        parser.add_argument('--warmup', default=200, type=int)
        return parent_parser

    def overlap_metrics(self, texts, versions, bert_h=None, bert_seq_lens=None):
        bert_tokens = list(map(self.clinbert_tokenizer.tokenize, texts))
        if bert_h is None or bert_seq_lens is None:
            bert_outputs, bert_seq_lens = encode(
                texts, self.clinbert_model, self.clinbert_tokenizer, self.device, max_length=128)
            bert_h = bert_outputs['hidden_states']
        # remove the CLS and final </s>
        bert_h_no_special = bert_h[:, 1:-1]

        context_toks = [t for version, t in zip(versions, bert_tokens) if version == 'context'][0]
        context_h = [h for version, h in zip(versions, bert_h_no_special) if version == 'context'][0]
        perturb_toks = [t for version, t in zip(versions, bert_tokens) if version == 'perturb']
        perturb_h = [h for version, h in zip(versions, bert_h_no_special) if version == 'perturb']
        generated_toks = [t for version, t in zip(versions, bert_tokens) if version == 'generated']
        generated_h = [h for version, h in zip(versions, bert_h_no_special) if version == 'generated']

        gold_toks = [t for version, t in zip(versions, bert_tokens) if version == 'gold']
        gold_h = [h for version, h in zip(versions, bert_h_no_special) if version == 'gold']
        gen_gold_bs_mean = gen_sim_improve = None
        gold_n = len(gold_toks)
        if gold_n > 0:
            assert gold_n == 1
            gold_toks = gold_toks[0]
            gold_h = gold_h[0]
            gen_gold_bs = [bertscore(g_tok, g_h, gold_toks, gold_h)[-1] for g_tok, g_h in zip(generated_toks, generated_h)]
            perturb_gold_bs = [bertscore(p_tok, p_h, gold_toks, gold_h)[-1] for p_tok, p_h in zip(perturb_toks, perturb_h)]
            gen_gold_bs_mean = np.mean(gen_gold_bs)
            perturb_gold_bs_mean = np.mean(perturb_gold_bs)
            gen_sim_improve = gen_gold_bs_mean - perturb_gold_bs_mean
        gen_perturb_bs = [bertscore(*x)[-1] for x in zip(generated_toks, generated_h, perturb_toks, perturb_h)]
        assert len(bert_h_no_special) == gold_n + 1 + len(perturb_h) + len(generated_h)
        perturb_context = [
            bertscore(p_tok, p_h, context_toks, context_h) for p_tok, p_h in zip(perturb_toks, perturb_h)]
        gen_context = [
            bertscore(g_tok, g_h, context_toks, context_h) for g_tok, g_h in zip(generated_toks, generated_h)]

        avg_perturb_context_prec = np.mean([x[0] for x in perturb_context])
        avg_perturb_context_cov = np.mean([x[1] for x in perturb_context])
        avg_perturb_context_f1 = np.mean([x[2] for x in perturb_context])

        avg_gen_context_prec = np.mean([x[0] for x in gen_context])
        avg_gen_context_cov = np.mean([x[1] for x in gen_context])
        avg_gen_context_f1 = np.mean([x[2] for x in gen_context])

        context_cov_improve = avg_gen_context_cov - avg_perturb_context_cov
        context_prec_improve = avg_gen_context_prec - avg_perturb_context_prec
        context_sim_improve = avg_gen_context_f1 - avg_perturb_context_f1

        gen_perturb_bs = np.mean(gen_perturb_bs)
        stats = {
            'gen_input_sim': gen_perturb_bs,
            'gen_target_sim': gen_gold_bs_mean,
            'gen_sim_improve': gen_sim_improve,
            'gen_context_cov_improve': context_cov_improve,
            'gen_context_prec_improve': context_prec_improve,
            'gen_context_sim_improve': context_sim_improve,
            'gen_context_cov': avg_gen_context_cov,
            'gen_context_prec': avg_gen_context_prec,
            'gen_context_f1': avg_gen_context_f1,
        }
        return stats

    def shared_step(self, batch, is_train):
        prefix = 'train' if is_train else 'val'
        mask_idxs = batch.pop('mask_idxs', None)
        pos_idxs = batch.pop('pos_idxs', None)
        neg_idxs = batch.pop('neg_idxs', None)
        labels = batch['labels']

        output = self.bart(**batch)
        # output = self.bart(**batch, output_attentions=True)
        # cross_attentions = output['cross_attentions'][-1].mean(dim=1)
        logits = output['logits']
        combined_loss = 0.0
        if mask_idxs is not None:
            log_probs_masked = torch.log_softmax(logits[mask_idxs], dim=-1)
            mask_lm_loss, _ = label_smoothed_nll_loss(log_probs_masked, labels[mask_idxs], reduce=True)
            self.log(f'{prefix}_lm_loss', mask_lm_loss, on_epoch=not is_train, on_step=is_train, prog_bar=True)
            combined_loss += mask_lm_loss
        con_loss = None
        if pos_idxs is not None:
            log_probs_pos = torch.log_softmax(logits[pos_idxs], dim=-1)
            likelihood_smooth_loss, _ = label_smoothed_nll_loss(log_probs_pos, labels[pos_idxs], reduce=True)
            con_loss = likelihood_smooth_loss
            self.log(
                f'{prefix}_like_loss', likelihood_smooth_loss, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        if neg_idxs is not None:
            probs_neg = torch.softmax(logits[neg_idxs], dim=-1)
            unlikelihood_smooth_loss, _ = label_smoothed_unlikelihood(probs_neg, labels[neg_idxs], reduce=True)
            con_loss += unlikelihood_smooth_loss
            self.log(
                f'{prefix}_unlike_loss', unlikelihood_smooth_loss, on_epoch=not is_train, on_step=is_train,
                prog_bar=True
            )
        if con_loss is not None:
            combined_loss += con_loss
            self.log(f'{prefix}_con_loss', con_loss, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{prefix}_loss', combined_loss, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        return combined_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, is_train=True)

    def generate(self, **kwargs):
        return self.bart.generate(**kwargs)

    def shared_generate(self, batch, num_beams=1):
        kwargs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'use_cache': True,
            'num_beams': num_beams,
            'max_length': self.hparams.max_output_length,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
        }

        generated_ids = self.bart.generate(**kwargs)
        generated_strs = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=False)
        input_strs = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=False)
        perturb_strs = []
        perturb_strs_clean = []
        context_str = input_strs[0][int(re.search(r'<sep>', input_strs[0]).end()):].replace('</s>', '').strip()

        for input_str in input_strs:
            perturb_sep = int(re.search(r'<sep>', input_str).start())
            perturb_str = input_str[:perturb_sep].replace('<s>', '').strip()
            perturb_strs.append(perturb_str)
            perturb_strs_clean.append(remove_prefix_and_tags(perturb_str))
        output_ids = batch['labels']
        output_ids[torch.where(batch['labels'] == -100)] = 1
        gold_strs = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=False)
        gold_str_clean = remove_prefix_and_tags(gold_strs[0])
        context_str_clean = remove_prefix_and_tags(context_str)
        generated_strs_clean = list(map(remove_prefix_and_tags, generated_strs))

        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        rouge_output = self.rouge.compute(predictions=generated_strs, references=gold_strs, rouge_types=rouge_types)

        all_texts = [gold_str_clean, context_str_clean] + perturb_strs_clean + generated_strs_clean
        versions = ['gold', 'context'] + ['perturb'] * len(perturb_strs) + ['generated'] * len(generated_strs)
        stats = self.overlap_metrics(all_texts, versions)

        f1s = []
        for rouge_type in rouge_types:
            stats[f'{rouge_type}_precision'] = rouge_output[rouge_type].mid.precision
            stats[f'{rouge_type}_recall'] = rouge_output[rouge_type].mid.recall
            stats[f'{rouge_type}_f1'] = rouge_output[rouge_type].mid.fmeasure
            f1s.append(rouge_output[rouge_type].mid.fmeasure)
        stats['mean_f1'] = np.array(f1s).mean()

        return generated_strs, gold_strs, stats

    def validation_step(self, batch, batch_idx):
        combined_loss = self.shared_step(batch, is_train=False)
        gen_idxs = batch.pop('pos_idxs', [1])
        agg_metrics = defaultdict(list)
        for pos_idx in gen_idxs:
            mini_batch = {k: v[[pos_idx]] for k, v in batch.items()}
            _, _, metrics = self.shared_generate(mini_batch)
            for k, v in metrics.items():
                agg_metrics[k].append(v)
        for k, v in agg_metrics.items():
            self.log(k, np.mean(v), on_epoch=True, on_step=False, prog_bar=True)

        return combined_loss

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
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup, num_training_steps=self.hparams.max_steps)

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items
