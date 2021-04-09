import logging
from collections import Counter
from typing import Dict, List, Tuple, Iterator

import numpy as np

import fca
import fca.algorithms
from tqdm import tqdm

import ptlm_wsid.target_context as tc

local_logger = logging.getLogger(__name__)


def fca_cluster(doc2preds: Dict[str, List[str]],
                n_sense_indicators=5,
                min_size=3,
                logger=None) -> List[fca.Concept]:
    if logger is None:
        logger = local_logger

    def get_cxt():
        intents = []
        objs = []
        for i, (docname, preds) in enumerate(doc2preds.items()):
            preds = set(preds)
            intents.append(preds)
            objs.append(docname)
        atts = list({x for intent in intents for x in intent})
        table = [[att in intent for att in atts] for intent in intents]
        cxt = fca.Context(cross_table=table, objects=objs, attributes=atts)
        return cxt

    def filter_cxt(cxt):
        # Remove infrequent attributes from the context to speed up the computations
        logger.debug(f'Original number of attributes: {len(cxt.attributes)}')
        # att_th = max(3, np.log10(len(cxt.objects)))
        att_th = np.log10(len(cxt.objects)) + 1
        logger.debug(f'Att extent threshold: {att_th}')
        att_del = []
        for att in cxt.attributes:
            if len(cxt.get_attribute_extent(att)) < att_th:
                att_del.append(att)
        cxt.delete_attributes(att_del)
        logger.debug(f'Number of attributes after cleaning: {len(cxt.attributes)}')
        cxt = cxt.clarify_objects()
        obj_del = []
        for obj in cxt.objects:
            obj_int = cxt.get_object_intent(obj)
            if len(obj_int) == 0 or len(obj_int) == len(cxt.attributes):
                # cxt.delete_object(obj)
                obj_del.append(obj)
        for obj in obj_del:
            cxt.delete_object(obj)
        return cxt

    def get_sense(factor):
        docs = list(factor.extent)
        term_rank = dict()
        common_terms = factor.intent
        for term in common_terms:
            term_rank[term] = sum(doc2preds[doc].index(term) for doc in docs)
        sense_ids = sorted(term_rank, key=lambda x: term_rank[x])[:n_sense_indicators]
        sense = fca.Concept(docs, set(sense_ids))
        return sense

    cxt = get_cxt()
    cxt = filter_cxt(cxt)
    chosen_factors = []
    chosen_scores = []
    chosen_senses = []
    factors_iter = fca.algorithms.factors.algorithm2_w_condition(
        cxt, fidelity=1, allow_repeatitions=False,
        min_atts_and_objs=min_size, objs_ge_atts=False
    )
    for i, (factor, factor_score, agg_score) in enumerate(factors_iter):
        cluster_threshold = sum(chosen_scores) / (len(chosen_scores) + 3) if chosen_scores else 0
        logger.debug(f'### NEXT FACTOR ###')
        logger.debug(f'Factor # {i}, score: {factor_score}')
        logger.debug(
            f'Extent: {len(factor.extent)}, intent: {len(factor.intent)}')
        if factor_score < cluster_threshold:
            break
        new_sense = get_sense(factor)
        logger.debug('Sense added.')
        chosen_factors.append(factor)
        chosen_scores.append(factor_score)
        chosen_senses.append(new_sense)
    logger.debug(f'Factors chosen: {len(chosen_factors)}')
    logger.debug('\nFactor #'.join([f'{i}: ' + str(factor.extent) + '\n' + str(factor.intent)
                                   for i, factor in enumerate(chosen_factors)]))
    return chosen_senses


def iter_substitutes(contexts, target_start_end_tuples,
                     titles=None,
                     top_n=50,
                     target_pos='N',
                     lang='deu',
                     th_substitute_len=3) -> Iterator[Tuple[str, List[str], List[str]]]:
    if titles is None:
        titles = range(len(contexts))
    cxt_targetinds = zip(contexts, target_start_end_tuples)
    top_n_per_pred = round(top_n/2)
    for i, (text_cxt, target_start_end) in enumerate(cxt_targetinds):
        cxt = tc.TargetContext(text_cxt, target_start_end)
        top_pred_unmasked = cxt.get_topn_predictions(top_n=top_n_per_pred,
                                                     target_pos=target_pos,
                                                     lang=lang,
                                                     do_mask=False,
                                                     th_len=th_substitute_len)
        top_pred_masked = cxt.get_topn_predictions(top_n=top_n_per_pred,
                                                   target_pos=target_pos,
                                                   lang=lang,
                                                   do_mask=True,
                                                   th_len=th_substitute_len)
        # top_pred = top_pred_unmasked + top_pred_masked
        yield titles[i] if (titles and len(titles) >= i) else i, top_pred_unmasked, top_pred_masked


def induce(contexts: List[str],
           target_start_end_tuples: List[Tuple[int, int]],
           titles: List[str] = None,
           target_pos: str = None,
           n_sense_descriptors=5, lang='eng', top_n_pred=100,
           min_number_contexts_for_fca_clustering=3, min_sub_len=3,
           verbose=False,
           logger=None) -> List[fca.Concept]:
    """
    The function induces sense(s) of the target from a collection of contexts.
    This function always returns a result. If the proper clustering does not
    produce any factors then the most common predictions are output.

    :param min_sub_len: min length of produced substitute
    :param contexts: the contexts themselves
    :param target_start_end_tuples: the (start index, end index) pairs
        indicating the target in the respective context.
        len(contexts) == len(target_start_end_tuples)
    :param top_n_pred: how many predictions are produced for each context
    :param titles: Titles of contexts
    :param n_sense_descriptors: how many sense indicators - subset of all
        predictions - are output for each sense
    :param target_pos: the desired part of speach of predictions
    :param lang: language. Used for POS tagging and lemmatization of predictions
    :param min_number_contexts_for_fca_clustering: minimum number of contexts
        to try the fca clustering. If there are only 1 or 2 then it often does
        not make sense to cluster.
    """
    if logger is None:
        logger = local_logger
    if not len(contexts) == len(target_start_end_tuples):
        raise ValueError(f'Length of contexts {len(contexts)} is not equal to '
                         f'the length of start and end indices list '
                         f'{len(target_start_end_tuples)}.')

    subs = iter_substitutes(
        contexts, target_start_end_tuples,
        titles=titles, th_substitute_len=min_sub_len,
        top_n=top_n_pred, target_pos=target_pos, lang=lang, )
    if verbose:
        subs = tqdm(subs, total=len(contexts))
    predicted = {title: top_pred_m + top_pred_unm
                 for title, top_pred_m, top_pred_unm in subs}

    senses = []
    target_phrase_in_fiurst_cintext = contexts[0][target_start_end_tuples[0][0]:target_start_end_tuples[0][1]]
    if len(contexts) >= min_number_contexts_for_fca_clustering:
        senses = fca_cluster(predicted,
                             n_sense_indicators=n_sense_descriptors)
        logger.debug(f'For {target_phrase_in_fiurst_cintext} with {len(contexts)} contexts '
                     f'fca_cluster produced {len(senses)} senses.')
    if not senses:  # fca_cluster did not produce results
        all_predicted = sum(predicted.values(), [])
        top_predicted = [x[0] for x in Counter(all_predicted).most_common(top_n_pred)]
        senses = [fca.Concept(intent=top_predicted,
                              extent=list(predicted.keys()))]
        logger.debug(f'For {target_phrase_in_fiurst_cintext} with {len(contexts)} contexts '
                     f'most common {len(top_predicted)} predictions are '
                     f'taken as sense indicators.')
    return senses
