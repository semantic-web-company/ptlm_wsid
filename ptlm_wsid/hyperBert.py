from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EvalPrediction
from transformers import Trainer, TrainingArguments

from scripts.wic_tsv import read_wic_tsv_ds

# model_name = 'bert-base-uncased'
# tok = AutoTokenizer.from_pretrained(model_name)


class HyperBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.classifier = nn.Linear(3*config.hidden_size, 1)  # BERT
        self.dropout = nn.Dropout(2*config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            target_start_len=None,
            descr_start_len=None,
            labels=None,

            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        tgt_inds = []
        descr_inds = []
        tgt_embeds = []
        descr_embeds = []
        for row in target_start_len.split(1):
            row_inds = range(row[0, 0], row.sum())
            tgt_inds.append(list(row_inds))
        for row in descr_start_len.split(1):
            row_inds = range(row[0, 0], row.sum())
            descr_inds.append(list(row_inds))

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        cls_output = bert_output[1]  # (bs, dim)

        for i, seq_out in enumerate(hidden_state.split(1, dim=0)):
            seq_out = seq_out.squeeze()
            row_tgt_embeds = seq_out[tgt_inds[i]]
            row_tgt_mean_embeds = torch.mean(row_tgt_embeds, dim=0).squeeze()  # (1, dim)
            row_descr_embeds = seq_out[descr_inds[i]]
            row_descr_mean_embeds = torch.mean(row_descr_embeds, dim=0).squeeze()  # (1, dim)
            tgt_embeds.append(row_tgt_mean_embeds)
            descr_embeds.append(row_descr_mean_embeds)
        target_output = torch.stack(tgt_embeds)  # (bs, dim)
        descr_output = torch.stack(descr_embeds)  # (bs, dim)
        pooled_output = torch.cat((
            target_output,
            descr_output,
            cls_output
        ), 1)  # (bs, 3*dim)
        # pooled_output = cls_output

        # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, 1)

        outputs = (logits,)  # + bert_output[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(1), labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits,  # (hidden_states), (attentions)


class WiCTSVDataset(torch.utils.data.Dataset):
    def __init__(self, contexts, target_inds, hypernyms, definitions, labels=None, focus_token='$'):
        self.len = len(contexts)
        self.labels = labels
        if focus_token is not None:
            prep_cxts = []
            prep_tgt_inds = []
            for cxt, tgt_ind in zip(contexts, target_inds):
                prep_cxt = cxt.split(' ')
                prep_cxt.insert(tgt_ind + 1, focus_token)  # after target
                prep_cxt.insert(tgt_ind, focus_token)  # before target
                prep_tgt_ind = tgt_ind + 1
                prep_cxts.append(' '.join(prep_cxt))
                prep_tgt_inds.append(prep_tgt_ind)
        else:
            prep_cxts = contexts
            prep_tgt_inds = target_inds
        self.encodings = tok([[context, definition + ' ; ' + f' {focus_token} '.join(hyps)]
                              for context, tgt_ind, definition, hyps in zip(prep_cxts, prep_tgt_inds, definitions, hypernyms)],
                             return_tensors='pt', truncation=True, padding=True)
        real_targets = [cxt.split(' ')[tgt_ind] for cxt, tgt_ind in zip(contexts, target_inds)]
        assert all(rt == p_cxt.split(' ')[p_ind] for rt, p_cxt, p_ind in zip(real_targets, prep_cxts, prep_tgt_inds))
        # contexts = [tok.tokenize(cxt) for cxt in contexts]
        self.tgt_start_len = []
        self.descr_start_len = []

        for cxt, tgt_ind, def_, hyps, real_tgt in zip(prep_cxts, prep_tgt_inds, definitions, hypernyms, real_targets):
            original_tokens = cxt.split(' ')

            index_list = []
            wps = []
            for original_index in range(len(original_tokens)):
                toked = tok.tokenize(original_tokens[original_index])
                index_list += [original_index] * len(toked)
                wps += toked
            index_map = defaultdict(list)
            for bert_index, original_index in enumerate(index_list):
                index_map[original_index].append(bert_index)

            # sanity check
            bert_tokens = tok.tokenize(cxt)
            assert len(bert_tokens) == len(sum(index_map.values(), [])), (bert_tokens, index_map)

            wp_tgt_ind = index_map[tgt_ind]  # index of target in the list of context's word pieces
            len_wp_tgt = len(wp_tgt_ind)
            wp_tgt_ind = wp_tgt_ind[0] + len(['[CLS]'])  # 1 for [CLS] token inserted later
            calc_tgt = wps[wp_tgt_ind - 1:wp_tgt_ind - 1 + len_wp_tgt]
            real_tgt = tok.tokenize(real_tgt)
            # sanity check
            assert calc_tgt == real_tgt, (calc_tgt, real_tgt, cxt)
            self.tgt_start_len.append((wp_tgt_ind, len_wp_tgt))
            descrs_start_ind = len(index_list) + len({'[CLS]', '[SEP]'})
            descrs_toks = tok.tokenize(def_ + ' ; ' + f' {focus_token} '.join(hyps))
            # descrs_toks = tok.tokenize(def_)
            descrs_len = len(descrs_toks)
            self.descr_start_len.append((descrs_start_ind, descrs_len))

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['target_start_len'] = torch.tensor(self.tgt_start_len[idx])
        item['descr_start_len'] = torch.tensor(self.descr_start_len[idx])
        if self.labels is not None:
            item['labels'] = torch.tensor(float(self.labels[idx]))
        return item

    def __len__(self):
        return self.len


if __name__ == '__main__':
    import os
    import logging
    logging.basicConfig(level=logging.INFO)

    base_path = Path(os.getenv('DATASET_PATH'))
    wic_tsv_train = base_path / 'Training'
    wic_tsv_dev = base_path / 'Development'
    wic_tsv_test = base_path / 'Test'
    output_path = Path(os.getenv('MODEL_OUTPUT_PATH'))

    model_name = 'bert-base-uncased'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = HyperBERT.from_pretrained(model_name)

    contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv_ds(wic_tsv_train)
    pos_sample_weight = (len(labels) - sum(labels)) / sum(labels)
    print('train', pos_sample_weight, Counter(labels))
    train_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions, labels)

    contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv_ds(wic_tsv_dev)
    pos_sample_weight = (len(labels) - sum(labels)) / sum(labels)
    print('dev', pos_sample_weight, Counter(labels))
    dev_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions, labels)

    contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv_ds(wic_tsv_test)
    pos_sample_weight = (len(labels) - sum(labels)) / sum(labels)
    print('test', pos_sample_weight, Counter(labels))
    test_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions, labels)

    def compute_metrics(p: EvalPrediction) -> Dict:
        fp = p.predictions
        print(fp[:10], fp[-10:])
        binary_preds = (p.predictions > 0).astype(type(p.label_ids[0]))
        preds: np.ndarray
        acc = (binary_preds == p.label_ids).mean()
        precision, r, f1, _ = precision_recall_fscore_support(y_true=p.label_ids, y_pred=binary_preds, average='binary')
        return {
            "acc": acc,
            "F_1": f1,
            "P": precision,
            "R": r,
            "Positive": binary_preds.sum() / binary_preds.shape[0]
        }

    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        # do_predict=True,
        num_train_epochs=5,
        per_device_train_batch_size=19,
        # per_gpu_train_batch_size=64,
        eval_steps=10,
        # save_steps=3000,
        # logging_first_step=True,
        logging_steps=10,
        # learning_rate=3e-5,
        # weight_decay=3e-7,
        # adam_epsilon=1e-7,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        # prediction_loss_only=True,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        # eval_dataset=test_ds,
        compute_metrics=compute_metrics,

        # tb_writer=writer
    )
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    output = trainer.train()
    print(f'Training output: {output}')
    trainer.save_model()
    preds = trainer.predict(test_dataset=test_ds)
    print(preds)
    print(preds.predictions.tolist())
