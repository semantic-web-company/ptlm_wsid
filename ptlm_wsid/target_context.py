import logging
import re
from typing import List, Tuple
import os

import torch
from nltk import sent_tokenize
from nltk.corpus import stopwords

from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
bert_model_str = os.getenv('BERT_MODEL', value='bert-base-uncased')  # 'bert-base-uncased', 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model_str)
model = BertForMaskedLM.from_pretrained(bert_model_str)
model.eval()

word_embeddings = model.bert.embeddings.word_embeddings


class TargetContext:
    MASK = '[MASK]'

    def __init__(self, context: str, target_start_end_inds: Tuple[int, int],
                 sent_tokenizer=sent_tokenize):
        self.context = context
        self.target_start_end_inds = target_start_end_inds
        self.sent_tokenizer = sent_tokenizer
        self._tokenized_with_mask = None

    def tokenize(self, text:str):
        sents = self.sent_tokenizer(text)
        tokenized = ['[CLS]']
        tokenized += [token
                      for sent in sents
                      for token in tokenizer.tokenize(sent)] + ['[SEP]']
        return tokenized

    @property
    def tokenized_with_target(self):
        raise NotImplementedError

    @staticmethod
    def crop(tokens, target_index, token_limit=510):
        ans = tokens
        if len(ans) > token_limit:
            start_ind = max(0, int(target_index - token_limit / 2))
            end_ind = min(len(ans), int(target_index + token_limit / 2))
            ans = ans[start_ind:end_ind]
        return ans

    # PREDICTIONS #
    def predict_mask_in_tokenized_cxt(self, target=None):
        if target is None:
            target = self.MASK
        assert target in self.tokenized_with_target, \
            self.tokenized_with_target
        target_ind = self.tokenized_with_target.index(target)
        tks_tensor = torch.tensor(
            [tokenizer.convert_tokens_to_ids(self.tokenized_with_target)])
        with torch.no_grad():
            predictions = model(tks_tensor)
        pred = predictions[0, target_ind]
        return pred

    def get_topn_predictions(self, top_n=5, lang='english'):
        pred = self.predict_mask_in_tokenized_cxt()
        top_predicted = torch.argsort(pred, descending=True)
        top_predicted = top_predicted.tolist()
        predicted_tokens = tokenizer.convert_ids_to_tokens(top_predicted)

        n = 0
        stopwords_set = set(stopwords.words(lang))
        ind_to_be_removed = []
        for i, token in enumerate(predicted_tokens):
            if len(token) < 3 or token.lower() in stopwords_set or \
                    token.startswith('##'):
                ind_to_be_removed.append(i)
            else:
                n += 1
                if n == top_n:
                    break
        for ind in ind_to_be_removed[::-1]:
            del top_predicted[ind]
        topn_pred = top_predicted[:top_n]
        predicted_tokens = tokenizer.convert_ids_to_tokens(topn_pred)
        logit_scores = [x.item() for x in pred[topn_pred]]
        return predicted_tokens, logit_scores

    @staticmethod
    def self_count_overlaps(target_ls: List[str], clusters: List[List[str]]):
        clusters_sets = [set(cl) for cl in clusters]
        target_set = set(target_ls)
        r = [len(target_set & c)/len(c) for c in clusters_sets]
        sum_r = sum(r)
        if sum_r:
            r_normed = [float(i) / sum_r for i in r]
            return r_normed
        else:
            return [1/len(r)]*len(r)

    def disambiguate(self, sense_clusters: List[List[str]],
                     top_n: int = None) -> List[float]:
        if top_n is None:
            top_n = len(sense_clusters[0])
        sense_embeddings = [embed(s_descrs) for s_descrs in sense_clusters]
        preds = self.get_topn_predictions(top_n=top_n)[0]
        target_embedding = embed(preds)
        proxs = [torch.cosine_similarity(target_embedding, sense_emb).
                     data.tolist()[0]
                 for sense_emb in sense_embeddings]
        return proxs


def embed(tokens):
    tks_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    embeddings = word_embeddings(tks_tensor)
    ans = torch.mean(embeddings, dim=1)
    return ans


class SubstituteTargetContext(TargetContext):
    @property
    def tokenized_with_target(self):
        target_start, target_end = self.target_start_end_inds
        context_with_mask = (self.context[:target_start] + f' {self.MASK} ' +
                             self.context[target_end:])
        tokenized = self.tokenize(context_with_mask)
        target_ind = tokenized.index(self.MASK)
        if tokenized[target_ind - 1] in ['a', 'an']:
            del tokenized[target_ind - 1]
        tokenized = self.crop(
            tokens=tokenized, target_index=target_ind
        )
        return tokenized


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


def sense_similarities(senses, targets):
    def embed(tokens):
        tks_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
        embeddings = word_embeddings(tks_tensor)
        ans = torch.mean(embeddings, dim=1)
        return ans

    target_embeddings = torch.stack([embed(tokenizer.tokenize(target)) for target in targets])
    target_embedding = torch.mean(target_embeddings, dim=0)
    sense_embeddings = [embed(sense) for sense in senses]
    proxs = [torch.cosine_similarity(target_embedding, sense_emb).data.tolist()[0]
             for sense_emb in sense_embeddings]
    print()
    print(targets)
    print(f'Proximities: {[prox for prox in proxs]}')
    return proxs

