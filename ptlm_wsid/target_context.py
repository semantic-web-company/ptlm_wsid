import logging
import re
import time
from functools import lru_cache
from typing import List, Tuple
import os

import torch, torch.nn
import spacy
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from transformers import AutoModelForMaskedLM, AutoTokenizer

lemmatizer = WordNetLemmatizer()

logger = logging.getLogger(__name__)

# global bert_tok, bert
model_tok = None
model_mlm = None
model = None


class LazySpacyDict(dict):
    """
    Load spacy nlp model when it is requested for the first time. Next time
    return the loaded model.
    """
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


def load_model():
    """
    This function loads bert model and populates the global variable. This way
    we know only one BERT is loaded.
    """
    global model_tok, model_mlm, model
    if model is None:
        model_name_or_path = os.getenv('TRANSFORMER_MODEL',
                                       default='distilbert-base-multilingual-cased')
        model_tok = AutoTokenizer.from_pretrained(model_name_or_path)
        model_mlm = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        model = model_mlm.distilbert
        model_mlm.eval()


class TargetContext:
    MASK = '[MASK]'

    def __init__(self, context:str,
                 target_start_end_inds:Tuple[int, int],
                 sent_tokenizer=sent_tokenize):
        """

        :param context: the context string
        :param target_start_end_inds: the start and end indices of the target in
             the context string
        :param sent_tokenizer: used to split the context into sentences
        """
        # TODO: add language?
        self.context = context
        self.target_start_end_inds = target_start_end_inds
        self.sent_tokenizer = sent_tokenizer
        self.target_start, self.target_end = self.target_start_end_inds
        self.target_str = self.context[self.target_start:self.target_end]
        self._pred = dict()
        self._tokenized = dict()
        self._target_indices = dict()

    # @property
    def pred(self, do_mask=True):
        if do_mask not in self._pred:
            load_model()
            tokenized = self.tokenize(do_mask=do_mask)
            logger.debug(f'Tokens: {tokenized}, mask: {do_mask}')
            target_indices = self.target_indices(do_mask=do_mask)
            if not do_mask and target_indices[1] - target_indices[0] > 1:
                tokenized[target_indices[0]+1:target_indices[1]] = []
            assert all(token in model_tok.vocab for token in tokenized)
            tks_tensor = torch.tensor(model_tok.encode(tokenized)).unsqueeze(0)
            predictions = model_mlm(tks_tensor)
            self._pred[do_mask] = predictions[0, target_indices[0]]
        return self._pred[do_mask]

    # @staticmethod
    # @lru_cache(maxsize=32)
    # def _predict_tokenized(tokenized: Tuple[str], target_ind:int):
    #     load_model()
    #     tks_tensor = torch.tensor(model_tok.encode(tokenized)).unsqueeze(0)
    #     predictions = model_mlm(tks_tensor)
    #     pred = predictions[0, target_ind]
    #     return pred
    #
    # def get_substitute_score(self, substitute_str:str, do_mask=True):
    #     substitute_toks = model_tok.tokenize(substitute_str)
    #
    #     tokens_to_score = self.tokenized(do_mask=do_mask).copy()
    #     target_inds = self.target_indices
    #     if len(substitute_toks) > 1:
    #         logger.debug(f'The substitute tokens are more than one: '
    #                      f'{substitute_toks}.')
    #         tokens_to_score[target_inds[0]:target_inds[1]] = [self.MASK]
    #         for _ in range(len(substitute_toks) - 1):
    #             tokens_to_score.insert(target_inds[0], self.MASK)
    #         logger.debug(f'Resulting tokens: {tokens_to_score}')
    #
    #     running_score = 0
    #     target_ind = target_inds[0]
    #     for tok in substitute_toks:
    #         pred = self._predict_tokenized(tuple(tokens_to_score), target_ind)
    #         tok_bert_ind = bert_tok.convert_tokens_to_ids([tok])
    #         running_score += pred[tok_bert_ind].item()
    #         tokens_to_score[target_ind] = tok
    #         target_ind += 1
    #     running_score /= len(substitute_toks)
    #
    #     return running_score

    def target_indices(self, do_mask=True):
        self.tokenize(do_mask=do_mask)
        return self._target_indices[do_mask]

    # @property
    def tokenize(self, do_mask=True):
        if do_mask not in self._tokenized:
            load_model()
            context = self.context[:self.target_start] + f' {self.MASK} ' +\
                          self.context[self.target_end:]
            target_str = self.MASK

            sents = self.sent_tokenizer(context)
            # tokenized = ['[CLS]']
            tokenized = [token
                         for sent in sents
                         for token in model_tok.tokenize(sent)]
                        # + ['[SEP]']
            target_ind = tokenized.index(target_str)
            if tokenized[target_ind - 1] in ['a', 'an']:
                del tokenized[target_ind - 1]
            tokenized, target_ind = self.crop(tokens=tokenized,
                                              target_index=target_ind)
            if not do_mask:
                target_toks = model_tok.tokenize(self.target_str)
                tokenized[target_ind:target_ind+1] = target_toks
                assert self.MASK not in tokenized
                target_inds = [target_ind, target_ind+len(target_toks)]
            else:
                target_inds = [target_ind, target_ind+1]

            self._tokenized[do_mask] = tokenized
            self._target_indices[do_mask] = target_inds
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
                             target_pos: str = None, do_mask=True) -> List[str]:
        """

        :param top_n: number of top predictions in the output
        :param lang: language iso 3 letter code
        :param target_pos: the desired POS tag of the prediction. 'N', 'J', etc.
        :param do_mask: if the target word should be masked during predictions.
        :return: list of top_n predictions
        """
        load_model()
        t_start_function = time.time()
        lem_pos = {'N': NOUN, 'NN': NOUN, 'NOUN': NOUN, 'PRON': NOUN,
                   'NNP': NOUN,
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
        predicted_tokens = model_tok.convert_ids_to_tokens(top_predicted)
        logger.debug(f'Generation of predictions and their formatting took '
                     f'{time.time() - t_start_function:.3f} s.')

        stopwords_set = set(stopwords.words(stopword_lang))
        topn_pred = []
        predicted_token_index = None
        logger.debug(f'starting predictions '
                     f'{time.time() - t_start_function:.3f} s.')
        t_start_predictions = time.time()
        spacy_tot_time = 0
        for i, predicted_token in enumerate(predicted_tokens):
            if i > top_n*10: break
            t_start_token = time.time()
            if len(predicted_token) < 3 or \
                    predicted_token.lower() in stopwords_set or \
                    predicted_token.startswith('##') or \
                    predicted_token.startswith('['):
                pass
            else:
                tokens = nlp(predicted_token)
                token_pos = tokens[0].tag_
                if target_pos is None or token_pos.startswith(target_pos):
                    predicted_token = lemmatizer.lemmatize(
                        predicted_token, lem_pos[target_pos])
                    topn_pred.append(predicted_token)
                    if len(topn_pred) >= top_n:
                        break
                spacy_tot_time += time.time() - t_start_token

        logger.debug(f'looping through predictions took '
                     f'{time.time() - t_start_predictions:.3f} s.')
        logger.debug(f'Spacy total time: {spacy_tot_time:0.3f} s')
        logger.debug(f'{i} tokens seen, {top_n} chosen')
        logger.debug(f'The whole function execution took '
                     f'{time.time() - t_start_function:.3f} s.')
        return topn_pred

    def disambiguate(self, sense_clusters:List[List[str]]) \
            -> List[float]:
        """
        Disambiguate using the provided sense clusters.
        Each returned score is between 0 and 1 - cosine distance of some vectors
        """

        def get_contextualized_embedding(ids):
            outputs, hidden_states = model(ids, output_hidden_states=True)
            # out = torch.mean(torch.stack(hidden_states[-2:]), dim=0).squeeze()
            out = outputs.squeeze()
            # out = hidden_states[-2].squeeze()
            return out

        load_model()
        senses_scores = []
        with torch.no_grad():
            cosine = torch.nn.CosineSimilarity(dim=0)

            tokens = self.tokenize(do_mask=False)
            target_start, target_end = self.target_indices(do_mask=False)
            cxt_ids = model_tok.encode(tokens)
            cxt_ids = torch.tensor(cxt_ids).unsqueeze(0)
            cxt_embedding = get_contextualized_embedding(cxt_ids)
            target_embeddings = cxt_embedding[target_start:target_end, :]
            target_embedding = torch.mean(target_embeddings, dim=0).squeeze()

            for sc in sense_clusters:
                sc_scores = []
                for mw_term in sc:
                    mw_term_ids = torch.tensor(model_tok.encode(mw_term)).unsqueeze(0)
                    mw_term_outputs = get_contextualized_embedding(mw_term_ids)
                    mw_term_embedding = torch.mean(mw_term_outputs[1:-1, :],
                                                   dim=0)
                    mw_term_score = cosine(mw_term_embedding,
                                           target_embedding).item()
                    assert isinstance(mw_term_score, float), mw_term
                    sc_scores.append(mw_term_score)
                sc_score = sum(sc_scores) / len(sc_scores)
                senses_scores.append(sc_score)
                # hyp_predictions = get_predict_logits(hypernyms_embedding, vocab)
                # cxt_predictions = get_predict_logits(target_embedding, vocab)
        logger.debug(f'Sense clusters received: {sense_clusters}, '
                     f'scores: {senses_scores}')
        return senses_scores
