"""
A module for Class Hierarchy Induction = CHI
"""
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Dict, List, TextIO, Iterable
from io import StringIO

from conllu import parse_incr, TokenList
from tqdm import tqdm

import fca
import ptlm_wsid.generative_factors as gf
import ptlm_wsid.utils as ptlm_utils

O_tags = {'"', 'NE', 'O'}


def parse_conll(f: TextIO, n_tokens: int = -1) -> Iterator[TokenList]:
    data = parse_incr(
        f, fields=['form', 'tag'],
        field_parsers={'tag': lambda line, i: line[i].split('-')})
    i = 0
    for sent in data:
        for w in sent:
            i += 1
            if 0 < n_tokens < i:
                raise StopIteration
            yield w['form'], w['tag']


def collect_ners(forms, tags, tokens_window=35):
    """

    :param forms:
    :param tags:
    :param tokens_window:
    :return:
    """
    assert len(forms) == len(tags)
    ners = []
    contexts = []
    ner_tags = []
    start_ends = []

    prev_tag = [None, '']
    ner_form = []
    for i, (form, tag) in enumerate(zip(forms, tags)):
        tag_head = tag if isinstance(tag, str) else tag[0]
        if tag_head not in O_tags:
            if form.strip():  # no empty string
                ner_form.append(form)
                prev_tag = tag
        elif tag_head in O_tags:
            if ner_form:
                ner_str = ' '.join(ner_form)
                ners.append(ner_str)
                ner_tags.append(prev_tag[1] if len(prev_tag) > 1 else None)
                cxt_start_ind = i - tokens_window - len(ner_form)
                cxt_start_ind = cxt_start_ind if cxt_start_ind > 0 else 0
                cxt_before = ' '.join(forms[cxt_start_ind:(i - len(ner_form))])
                start_ind = len(cxt_before) + 1  #
                end_ind = start_ind + len(ner_str)
                ner_form = []
                cxt_after = ' '.join(forms[i+1:i+tokens_window])
                cxt = cxt_before + ' ' + ner_str + ' ' + cxt_after
                contexts.append(cxt)
                start_ends.append( (start_ind, end_ind) )
                assert ner_str == cxt[start_ind:end_ind], (ner_str, cxt[start_ind:end_ind])

    return ners, ner_tags, contexts, start_ends


def get_senses_per_ne(ner_agg: Dict[str, Iterable[int]],
                      contexts: List[str],
                      start_ends: List[Tuple[int, int]],
                      lang='deu', cxts_limit=50, n_pred=50, target_pos='N',
                      th_att_len=4) -> Dict[str, Iterable[str]]:
    ners_dict = dict()
    for n, (ner_form, ner_inds) in enumerate(tqdm(list(ner_agg.items()))):
        ner_cxts, ner_ses = list(zip(*[(contexts[i], start_ends[i])
                                       for i in ner_inds]))
        ner_senses = gf.induce(contexts=ner_cxts[:cxts_limit],
                               target_start_end_tuples=ner_ses[:cxts_limit],
                               target_pos=target_pos, n_sense_indicators=10,
                               lang=lang, top_n_pred=n_pred, verbose=False,
                               min_number_contexts_for_fca_clustering=5,
                               min_sub_len=th_att_len)

        if len(ner_senses) > 1:
            for i, sense in enumerate(ner_senses):
                ners_dict[f'{ner_form}##{i}'] = list(sense.intent)
        else:
            ners_dict[f'{ner_form}'] = list(ner_senses[0].intent)
    return ners_dict
