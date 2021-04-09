"""
A module for inducing senses of target phrases from a set of contexts
"""
from typing import Iterator, Tuple, Dict, List, Iterable

from tqdm import tqdm

import ptlm_wsid.generative_factors as gf

O_tags = {'"', 'NE', 'O'}


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
                start_ind = len(cxt_before) + 1  # compensate for additional space after cxt_before
                end_ind = start_ind + len(ner_str)
                ner_form = []
                cxt_after = ' '.join(forms[i:i+tokens_window])
                cxt = cxt_before + ' ' + ner_str + ' ' + cxt_after
                contexts.append(cxt)
                start_ends.append( (start_ind, end_ind) )
                assert ner_str == cxt[start_ind:end_ind], (ner_str, cxt[start_ind:end_ind])

    return ners, ner_tags, contexts, start_ends


def iter_senses(ner_agg: Dict[str, Iterable[int]],
                contexts: List[str],
                start_ends: List[Tuple[int, int]],
                lang='deu', cxts_limit=50, n_pred=50, target_pos='N',
                n_sense_descriptors=10, th_att_len=4,
                logger=None):
    """
    :param th_att_len: min length of produced substitute
    :param n_pred: how many predictions are produced for each context
    :param n_sense_descriptors: how many sense indicators - subset of all
        predictions - are output for each sense
    :param target_pos: the desired part of speach of predictions
    :param lang: language. Used for POS tagging and lemmatization of predictions.
    """
    pbar = tqdm(list(ner_agg.items()))
    total_examples = 0
    for ner_form, ner_inds in pbar:
        ner_cxts, ner_ses = list(zip(*[(contexts[i], start_ends[i])
                                       for i in ner_inds]))
        ner_senses = gf.induce(contexts=ner_cxts[:cxts_limit],
                               target_start_end_tuples=ner_ses[:cxts_limit],
                               target_pos=target_pos, lang=lang, verbose=False,
                               n_sense_descriptors=n_sense_descriptors,
                               top_n_pred=n_pred, min_sub_len=th_att_len,
                               min_number_contexts_for_fca_clustering=5,
                               logger=logger)
        total_examples += len(ner_cxts[:cxts_limit])
        pbar.set_description(desc=f'{total_examples} contexts processed')
        yield ner_form, [x.intent for x in ner_senses]
