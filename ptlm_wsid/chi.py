"""
A module for Class Hierarchy Induction = CHI
"""
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Dict, List, TextIO, Iterable
from io import StringIO

from conllu import parse, TokenList
from tqdm import tqdm

import fca
import ptlm_wsid.generative_factors as gf
import ptlm_wsid.utils as ptlm_utils

O_tags = {'"', 'NE', 'O'}


# def parse_conll(f: TextIO, fields=('form', 'tag'),
#                 n_tokens: int = -1) -> Iterator[TokenList]:
#     data = parse_incr(
#         f, fields=fields,
#         field_parsers={'tag': lambda line, i: line[i].split('-')})
#     i = 0
#     for sent in data:
#         for w in sent:
#             i += 1
#             if 0 < n_tokens < i:
#                 return
#             yield w['form'], w['tag']


def parse_conll(data_str: str, fields=('form', '1', '2', 'tag'),
                n_tokens: int = -1) -> Iterator[TokenList]:
    data = parse(data_str, fields=fields,
                 field_parsers={'tag': lambda line, i: line[i].split('-')})
    i = 0
    for sent in data:
        for w in sent:
            i += 1
            if 0 < n_tokens < i:
                return
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


def iter_senses(ner_agg: Dict[str, Iterable[int]],
                contexts: List[str],
                start_ends: List[Tuple[int, int]],
                lang='deu', cxts_limit=50, n_pred=50, target_pos='N',
                n_sense_indicators=10, th_att_len=4):
    for ner_form, ner_inds in tqdm(list(ner_agg.items())):
        ner_cxts, ner_ses = list(zip(*[(contexts[i], start_ends[i])
                                       for i in ner_inds]))
        ner_senses = gf.induce(contexts=ner_cxts[:cxts_limit],
                               target_start_end_tuples=ner_ses[:cxts_limit],
                               target_pos=target_pos, lang=lang, verbose=False,
                               n_sense_indicators=n_sense_indicators,
                               top_n_pred=n_pred, min_sub_len=th_att_len,
                               min_number_contexts_for_fca_clustering=5)
        yield ner_form, [x.intent for x in ner_senses]


# def get_senses_per_ne(ner_agg: Dict[str, Iterable[int]],
#                       contexts: List[str],
#                       start_ends: List[Tuple[int, int]],
#                       lang='deu', cxts_limit=50, n_pred=50, target_pos='N',
#                       th_att_len=4) -> Dict[str, Iterable[str]]:
#     ners_dict = dict()
#     senses_iterator = iter_senses(ner_agg, contexts, start_ends,
#                                   lang=lang, cxts_limit=cxts_limit,
#                                   n_pred=n_pred, target_pos=target_pos,
#                                   th_att_len=th_att_len)
#     for ner_form, ner_senses in senses_iterator:
#         if len(ner_senses) > 1:
#             for i, sense in enumerate(ner_senses):
#                 ners_dict[f'{ner_form}##{i}'] = list(sense)
#         else:
#             ners_dict[f'{ner_form}'] = list(ner_senses[0])
#     return ners_dict
