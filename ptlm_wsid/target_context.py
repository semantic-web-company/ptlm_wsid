import logging
import re
import time
from functools import lru_cache
from typing import List, Tuple
import os

import torch
import spacy
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

lemmatizer = WordNetLemmatizer()

logger = logging.getLogger(__name__)

global bert_tok, bert
bert_tok = None
bert = None


class LazySpacyDict(dict):
    def __getitem__(self, item):
        if item.startswith('de') or item.startswith('ge'):
            if 'de' not in self:
                self['de'] = spacy.load('de_core_news_sm')
            return self.get('de'), 'german'
        elif item.startswith('en'):
            if 'en' not in self:
                self['en'] = spacy.load('en_core_web_sm')
            return self.get('en'), 'english'
        elif item.startswith('nl'):
            if 'nl' not in self:
                self['nl'] = spacy.load('nl_core_news_sm')
            return self.get('nl'), 'dutch'
        elif item.startswith('es') or item.startswith('sp'):
            if 'es' not in self:
                self['es'] = spacy.load('es_core_news_sm')
            return self.get('es'), 'spanish'
        else:
            assert 0, 'Only en, de, nl and es languages are supported so far.'


nlp_dict = LazySpacyDict()


def load_bert():
    global bert_tok, bert
    if bert is None:
        bert_model_str = os.getenv('BERT_MODEL', default='bert-base-multilingual-uncased')  # 'bert-base-uncased', 'bert-base-multilingual-uncased'
        bert_tok = BertTokenizer.from_pretrained(bert_model_str)
        bert = BertForMaskedLM.from_pretrained(bert_model_str)
        bert.eval()


class TargetContext:
    MASK = '[MASK]'

    def __init__(self, context:str, target_start_end_inds:Tuple[int, int],
                 sent_tokenizer=sent_tokenize):
        # TODO: add language?
        load_bert()
        self.context = context
        self.target_start_end_inds = target_start_end_inds
        self.sent_tokenizer = sent_tokenizer
        self.target_start, self.target_end = self.target_start_end_inds
        self.target_str = self.context[self.target_start:self.target_end]
        self._pred = dict()
        self._tokenized = dict()
        self._target_indices = None

    # @property
    def pred(self, do_mask=True):
        if do_mask not in self._pred:
            tokenized = self.tokenized(do_mask=do_mask)
            logger.debug(f'Tokens: {tokenized}, mask: {do_mask}')
            if not do_mask and \
                    self.target_indices[1] - self.target_indices[0] > 1:
                tokenized[self.target_indices[0]+1:self.target_indices[1]] = []
            assert all(token in bert_tok.vocab for token in tokenized)
            tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(tokenized)])
            with torch.no_grad():
                predictions = bert(tks_tensor)
            self._pred[do_mask] = predictions[0, self.target_indices[0]]
        return self._pred[do_mask]

    @staticmethod
    @lru_cache(maxsize=32)
    def _predict_tokenized(tokenized: Tuple[str], target_ind:int):
        tks_tensor = torch.tensor([bert_tok.convert_tokens_to_ids(tokenized)])
        with torch.no_grad():
            predictions = bert(tks_tensor)
        pred = predictions[0, target_ind]
        return pred

    def get_substitute_score(self, substitute_str:str, do_mask=True):
        substitute_toks = bert_tok.tokenize(substitute_str)

        tokens_to_score = self.tokenized(do_mask=do_mask).copy()
        target_inds = self.target_indices
        if len(substitute_toks) > 1:
            logger.debug(f'The substitute tokens are more than one: '
                         f'{substitute_toks}.')
            tokens_to_score[target_inds[0]:target_inds[1]] = [self.MASK]
            for _ in range(len(substitute_toks) - 1):
                tokens_to_score.insert(target_inds[0], self.MASK)
            logger.debug(f'Resulting tokens: {tokens_to_score}')

        running_score = 0
        target_ind = target_inds[0]
        for tok in substitute_toks:
            pred = self._predict_tokenized(tuple(tokens_to_score), target_ind)
            tok_bert_ind = bert_tok.convert_tokens_to_ids([tok])
            running_score += pred[tok_bert_ind].item()
            tokens_to_score[target_ind] = tok
            target_ind += 1
        running_score /= len(substitute_toks)

        return running_score

    @property
    def target_indices(self):
        self.tokenized()
        return self._target_indices

    # @property
    def tokenized(self, do_mask=True):
        if do_mask not in self._tokenized:
            context = self.context[:self.target_start] + f' {self.MASK} ' +\
                          self.context[self.target_end:]
            target_str = self.MASK

            sents = self.sent_tokenizer(context)
            tokenized = ['[CLS]']
            tokenized += [token
                          for sent in sents
                          for token in bert_tok.tokenize(sent)] + ['[SEP]']
            target_ind = tokenized.index(target_str)
            if tokenized[target_ind - 1] in ['a', 'an']:
                del tokenized[target_ind - 1]
            tokenized, target_ind = self.crop(tokens=tokenized,
                                              target_index=target_ind)
            if not do_mask:
                target_toks = bert_tok.tokenize(self.target_str)
                tokenized[target_ind:target_ind+1] = target_toks
                assert self.MASK not in tokenized
                target_inds = [target_ind, target_ind+len(target_toks)]
            else:
                target_inds = [target_ind, target_ind+1]

            self._tokenized[do_mask] = tokenized
            self._target_indices = target_inds
        return self._tokenized[do_mask]

    @staticmethod
    def crop(tokens, target_index, token_limit=300):
        ans = tokens
        if len(ans) > token_limit:
            start_ind = max(0, int(target_index - token_limit / 2))
            end_ind = min(len(ans), int(target_index + token_limit / 2))
            ans = ans[start_ind:end_ind]
            target_index = target_index - start_ind
        return ans, target_index

    def get_topn_predictions(self, top_n=10, lang='eng',
                             target_pos=None, do_mask=True):
        lem_pos = {'N': NOUN, 'NN': NOUN, 'NOUN': NOUN, 'PRON': NOUN,
                   'PROPN': NOUN, None: NOUN,
                   'J': ADJ, 'JJ': ADJ, 'ADJ': ADJ, 'A': ADJ,
                   'R': ADV, 'ADV': ADV,
                   'V': VERB, 'VERB': VERB,
                   'X': NOUN}
        lem_pos.setdefault(NOUN)
        global nlp_dict
        nlp, stopword_lang = nlp_dict[lang]

        if target_pos is None:
            doc = nlp(self.context)
            for doc_tok in doc:
                if str(doc_tok.text) == self.target_str:
                    target_pos = str(doc_tok.tag_)
                    break

        pred = self.pred(do_mask=do_mask)
        top_predicted = torch.argsort(pred, descending=True)
        top_predicted = top_predicted.tolist()
        predicted_tokens = bert_tok.convert_ids_to_tokens(top_predicted)

        stopwords_set = set(stopwords.words(stopword_lang))
        topn_pred = []
        predicted_token_index = None
        for i, predicted_token in enumerate(predicted_tokens):
            if len(predicted_token) < 3 or predicted_token.lower() in stopwords_set or \
                    predicted_token.startswith('##'):
                pass
            else:
                cxt_with_token = self.context[:self.target_start] + \
                                 f' {predicted_token} ' + self.context[self.target_end:]
                t_start = time.time()
                doc = nlp(cxt_with_token)
                logger.debug(f'spacy took {time.time() - t_start:.3f} s.')

                if predicted_token_index is None:
                    for i, doc_token in enumerate(doc):
                        if str(doc_token.text) == predicted_token:
                            predicted_token_index = i
                            token_pos = doc_token.tag_
                            break
                    else:
                        token_pos = target_pos
                else:
                    token_pos = str(doc[predicted_token_index].tag_)

                if target_pos is None or token_pos.startswith(target_pos):
                    predicted_token = lemmatizer.lemmatize(predicted_token, lem_pos[target_pos])
                    topn_pred.append(predicted_token)
                    if len(topn_pred) >= top_n:
                        break
        return topn_pred

    def disambiguate(self, sense_clusters:List[List[str]], do_mask=True) \
            -> List[float]:
        sense_scores = []
        for sense_cluster in sense_clusters:
            scores = dict()
            for indicator in sense_cluster:
                score = self.get_substitute_score(indicator, do_mask=do_mask)
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
