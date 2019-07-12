import logging
import re
from functools import lru_cache
from typing import List, Tuple
import os

import torch
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

lemmatizer = WordNetLemmatizer()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

global bert_tok, bert
bert_tok = None
bert = None


def load_bert():
    global bert_tok, bert
    if bert is None:
        bert_model_str = os.getenv('BERT_MODEL', default='bert-base-uncased')  # 'bert-base-uncased', 'bert-base-multilingual-uncased'
        bert_tok = BertTokenizer.from_pretrained(bert_model_str)
        bert = BertForMaskedLM.from_pretrained(bert_model_str)
        bert.eval()


class TargetContext:
    MASK = '[MASK]'

    def __init__(self, context:str, target_start_end_inds:Tuple[int, int],
                 sent_tokenizer=sent_tokenize):
        load_bert()
        self.context = context
        self.target_start_end_inds = target_start_end_inds
        self.sent_tokenizer = sent_tokenizer
        self.target_start, self.target_end = self.target_start_end_inds
        self.target_str = self.context[self.target_start:self.target_end]
        self._pred = None
        self._tokenized = None
        self._mask_index = None

    @property
    def pred(self):
        if self._pred is None:
            tokenized = self.tokenized
            assert all(token in bert_tok.vocab for token in tokenized)
            tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(tokenized)])
            with torch.no_grad():
                predictions = bert(tks_tensor)
            self._pred = predictions[0, self.mask_index]
        return self._pred

    @staticmethod
    @lru_cache(maxsize=32)
    def _predict_tokenized(tokenized: Tuple[str], target_ind:int):
        tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(tokenized)])
        with torch.no_grad():
            predictions = bert(tks_tensor)
        pred = predictions[0, target_ind]
        return pred

    def get_substitute_score(self, substitute_str:str):
        substitute_toks = bert_tok.tokenize(substitute_str)

        tokens_to_score = self.tokenized.copy()
        mask_ind = self.mask_index
        for _ in range(len(substitute_toks) - 1):
            tokens_to_score.insert(mask_ind, self.MASK)

        running_score = 0
        for tok in substitute_toks:
            pred = self._predict_tokenized(tuple(tokens_to_score), mask_ind)
            tok_bert_ind = bert_tok.convert_tokens_to_ids([tok])
            running_score += pred[tok_bert_ind].item()
            tokens_to_score[mask_ind] = tok
            mask_ind += 1
        running_score /= len(substitute_toks)

        return running_score

    @property
    def mask_index(self):
        self.tokenized
        return self._mask_index

    @property
    def tokenized(self):
        if self._tokenized is None:
            text_masked = self.context[:self.target_start] + f' {self.MASK} ' +\
                          self.context[self.target_end:]

            sents = self.sent_tokenizer(text_masked)
            tokenized = ['[CLS]']
            tokenized += [token
                          for sent in sents
                          for token in bert_tok.tokenize(sent)] + ['[SEP]']
            mask_ind = tokenized.index(self.MASK)
            if tokenized[mask_ind - 1] in ['a', 'an']:
                del tokenized[mask_ind - 1]
            tokenized = self.crop(tokens=tokenized,
                                  target_index=mask_ind)
            self._tokenized = tokenized
            self._mask_index = tokenized.index(self.MASK)
        return self._tokenized
        ########################

    @staticmethod
    def crop(tokens, target_index, token_limit=510):
        ans = tokens
        if len(ans) > token_limit:
            start_ind = max(0, int(target_index - token_limit / 2))
            end_ind = min(len(ans), int(target_index + token_limit / 2))
            ans = ans[start_ind:end_ind]
        return ans

    def get_topn_predictions(self, top_n=10, lang='english', target_pos='NN'):
        pred = self.pred
        top_predicted = torch.argsort(pred, descending=True)
        top_predicted = top_predicted.tolist()
        predicted_tokens = bert_tok.convert_ids_to_tokens(top_predicted)

        # n = 0
        stopwords_set = set(stopwords.words(lang))
        topn_pred = []
        for i, token in enumerate(predicted_tokens):
            if len(token) < 3 or token.lower() in stopwords_set or \
                    token.startswith('##'):
                pass
            else:
                cxt_with_token = self.context[:self.target_start] + \
                                 f' {token} ' + self.context[self.target_end:]
                cxt_toks = word_tokenize(text=cxt_with_token)
                toks_pos = pos_tag(cxt_toks)
                for tok, pos in toks_pos:
                    if tok == token:
                        token_pos = pos
                        break
                else:
                    token_pos = target_pos
                if token_pos.startswith(target_pos):
                    token = lemmatizer.lemmatize(token, 'n')
                    topn_pred.append(token)
                    if len(topn_pred) >= top_n:
                        break
        return topn_pred

    def disambiguate(self, sense_clusters:List[List[str]]) -> List[float]:
        sense_scores = []
        for sense_cluster in sense_clusters:
            scores = dict()
            for indicator in sense_cluster:
                score = self.get_substitute_score(indicator)
                scores[indicator] = score

            # top_scores = sorted(scores.values(), reverse=True)[:topn]
            sense_score = sum(scores.values()) / len(scores) if scores else 0
            sense_scores.append(sense_score)
        return sense_scores


class Document:
    TARGET = '[TARGET]'
    MASK = '[MASK]'

    def __init__(self, name, text):
        self.name = name
        self.text = text
        self._sent_tokenized = None

    @property
    def sent_tokenized(self):
        if self._sent_tokenized is None:
            self._sent_tokenized = sent_tokenize(self.text)
        return self._sent_tokenized

    def sub_targets(self, target_label):
        tp = self.target_pattern(target_label)
        subed = tp.sub(' ' + self.MASK + ' ', self.text)
        return subed

    @staticmethod
    def target_pattern(target_label):
        rc = re.compile(r'(?<=\b){}\w{{0,2}}(?=\b)'.format(target_label),
                        re.IGNORECASE)
        return rc

    def extract_target_cxts(self, target_labels: List[str]) -> \
            List[Tuple[str, Tuple[int, int]]]:
        target_cxts = []
        tps = [self.target_pattern(tl) for tl in target_labels]
        for i, sent in enumerate(self.sent_tokenized):
            ms = [tp.search(sent) for tp in tps]
            for m in ms:
                if m:
                    if len(sent) < 20:
                        target_sents = (self.sent_tokenized[i-1] + sent +
                                        self.sent_tokenized[i+1])
                    else:
                        target_sents = sent
                    match_start_end = (m.start(), m.end())
                    target_cxts.append((target_sents, match_start_end))
                    break
        return target_cxts

##################################################
# def sense_similarities(senses, targets):
#     def embed(tokens):
#         tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(tokens)])
#         embeddings = word_embeddings(tks_tensor)
#         ans = torch.mean(embeddings, dim=1)
#         return ans
#
#     target_embeddings = torch.stack([embed(bert_tok.tokenize(target)) for target in targets])
#     target_embedding = torch.mean(target_embeddings, dim=0)
#     sense_embeddings = [embed(sense) for sense in senses]
#     proxs = [torch.cosine_similarity(target_embedding, sense_emb).data.tolist()[0]
#              for sense_emb in sense_embeddings]
#     print()
#     print(targets)
#     print(f'Proximities: {[prox for prox in proxs]}')
#     return proxs
#
# class TargetContext:
#     MASK = '[MASK]'
#
#     def __init__(self, context: str, target_start_end_inds: Tuple[int, int],
#                  sent_tokenizer=sent_tokenize):
#         self.context = context
#         self.target_start_end_inds = target_start_end_inds
#         self.sent_tokenizer = sent_tokenizer
#         self._tokenized_with_mask = None
#         target_start, target_end = self.target_start_end_inds
#         self.target_str = self.context[target_start:target_end]
#
#     def tokenize(self, text:str=None, target_start_end:Tuple[int, int]=None):
#         if text is None:
#             text = self.context
#         if target_start_end is None:
#             target_start_end = self.target_start_end_inds
#         target_start, target_end = target_start_end
#
#         # sents = self.sent_tokenizer(text)
#         # tokenized = ['[CLS]']
#         # tokenized += [token
#         #               for sent in sents
#         #               for token in bert_tok.tokenize(sent)] + ['[SEP]']
#
#         before_text = text[:target_start]
#         before_sents = self.sent_tokenizer(before_text)
#         before_tokenized = ['[CLS]']
#         before_tokenized += [token
#                              for sent in before_sents
#                              for token in bert_tok.tokenize(sent)]
#
#         target_toks = bert_tok.tokenize(text[target_start:target_end])
#         target_tok_start = len(before_tokenized)
#         target_tok_end = target_tok_start + len(target_toks)
#         tokenized = before_tokenized + target_toks + [
#             token
#             for sent in self.sent_tokenizer(text[target_end:])
#             for token in bert_tok.tokenize(sent)]
#
#         return tokenized, (target_tok_start, target_tok_end)
#
#     @property
#     def tokenized_with_target(self):
#         raise NotImplementedError
#
#     @staticmethod
#     def crop(tokens, target_index, token_limit=510):
#         ans = tokens
#         if len(ans) > token_limit:
#             start_ind = max(0, int(target_index - token_limit / 2))
#             end_ind = min(len(ans), int(target_index + token_limit / 2))
#             ans = ans[start_ind:end_ind]
#         return ans
#
#     # PREDICTIONS #
#     def predict_mask_in_tokenized_cxt(self, target=None):
#         if target is None:
#             target = self.MASK
#         assert target in self.tokenized_with_target, \
#             self.tokenized_with_target
#         target_ind = self.tokenized_with_target.index(target)
#         tks_tensor = torch.tensor(
#             [bert_tok.convert_tokens_to_ids(self.tokenized_with_target)])
#         with torch.no_grad():
#             predictions = bert(tks_tensor)
#         pred = predictions[0, target_ind]
#         return pred
#
#     def get_topn_predictions(self, top_n=5, lang='english'):
#         pred = self.predict_mask_in_tokenized_cxt()
#         top_predicted = torch.argsort(pred, descending=True)
#         top_predicted = top_predicted.tolist()
#         predicted_tokens = bert_tok.convert_ids_to_tokens(top_predicted)
#
#         n = 0
#         stopwords_set = set(stopwords.words(lang))
#         ind_to_be_removed = []
#         for i, token in enumerate(predicted_tokens):
#             if len(token) < 3 or token.lower() in stopwords_set or \
#                     token.startswith('##'):
#                 ind_to_be_removed.append(i)
#             else:
#                 n += 1
#                 if n == top_n:
#                     break
#         for ind in ind_to_be_removed[::-1]:
#             del top_predicted[ind]
#         topn_pred = top_predicted[:top_n]
#         predicted_tokens = bert_tok.convert_ids_to_tokens(topn_pred)
#         logit_scores = [x.item() for x in pred[topn_pred]]
#         return predicted_tokens, logit_scores
#
#     @staticmethod
#     def self_count_overlaps(target_ls: List[str], clusters: List[List[str]]):
#         clusters_sets = [set(cl) for cl in clusters]
#         target_set = set(target_ls)
#         r = [len(target_set & c)/len(c) for c in clusters_sets]
#         sum_r = sum(r)
#         if sum_r:
#             r_normed = [float(i) / sum_r for i in r]
#             return r_normed
#         else:
#             return [1/len(r)]*len(r)
#
#     def disambiguate(self, sense_clusters: List[List[str]],
#                      top_n: int = None) -> List[float]:
#         if top_n is None:
#             top_n = len(sense_clusters[0])
#         sense_embeddings = [embed(s_descrs) for s_descrs in sense_clusters]
#         preds = self.get_topn_predictions(top_n=top_n)[0]
#         target_embedding = embed(preds)
#         proxs = [torch.cosine_similarity(target_embedding, sense_emb).
#                      data.tolist()[0]
#                  for sense_emb in sense_embeddings]
#         return proxs
#
#
# def embed(tokens):
#     tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(tokens)])
#     embeddings = word_embeddings(tks_tensor)
#     ans = torch.mean(embeddings, dim=1)
#     return ans
#
#
# class SubstituteTargetContext(TargetContext):
#     @property
#     def tokenized_with_target(self):
#         target_start, target_end = self.target_start_end_inds
#         context_with_mask = (self.context[:target_start] + f' {self.MASK} ' +
#                              self.context[target_end:])
#         tokenized = self.tokenize(context_with_mask)
#         target_ind = tokenized.index(self.MASK)
#         if tokenized[target_ind - 1] in ['a', 'an']:
#             del tokenized[target_ind - 1]
#         tokenized = self.crop(
#             tokens=tokenized, target_index=target_ind
#         )
#         return tokenized
# word_embeddings = bert.bert.embeddings.word_embeddings
#
#
# def bert_predict_top_n(toks, target_ind, top_n=25, lang='english'):
#     tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(toks)])
#     with torch.no_grad():
#         predictions = bert(tks_tensor)
#     pred_ind = predictions[0, target_ind]
#
#     top_predicted = torch.argsort(pred_ind, descending=True)
#     top_predicted = top_predicted.tolist()
#     predicted_tokens = bert_tok.convert_ids_to_tokens(top_predicted)
#
#     n = 0
#     stopwords_set = set(stopwords.words(lang))
#     ind_to_be_removed = []
#     for i, token in enumerate(predicted_tokens):
#         if len(token) < 3 or token.lower() in stopwords_set or \
#                 token.startswith('##'):
#             ind_to_be_removed.append(i)
#         else:
#             n += 1
#             if n == top_n:
#                 break
#     for ind in ind_to_be_removed[::-1]:
#         del top_predicted[ind]
#     topn_pred = top_predicted[:top_n]
#     predicted_tokens = bert_tok.convert_ids_to_tokens(topn_pred)
#     logit_scores = [x.item() for x in pred_ind[topn_pred]]
#     return predicted_tokens, logit_scores
#
#
# def bert_get_prob(token_seq, target_tokens, ind):
#     tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(token_seq)])
#     with torch.no_grad():
#         predictions = bert(tks_tensor)
#     pred = predictions[0, ind]
#
#     token_bert_ind = bert_tok.convert_tokens_to_ids([target_tokens])
#     return expit(pred[token_bert_ind].item())
#
# # TODO: probably cache?
# def _bert_get_predictions(tokenized, target_ind):
#     #  tokenized, (target_tok_start, target_tok_end) = self.tokenize()
#     # inst = TargetContext(context, target_start_end_inds)
#     # tokenized, (target_tok_start, target_tok_end) = inst.tokenize()
#     for token in tokenized:
#         if not token in bert_tok.vocab:
#             raise ValueError(f'Token "{token}" not in the tokenizer vocabulary.')
#
#     tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(tokenized)])
#     with torch.no_grad():
#         predictions = bert(tks_tensor)
#     pred = predictions[0, target_ind]
#
#     top_predicted = torch.argsort(pred, descending=True)
#     top_predicted = top_predicted.tolist()
#     predicted_tokens = bert_tok.convert_ids_to_tokens(top_predicted)
#
#     return pred
