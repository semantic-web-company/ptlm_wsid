import logging

import numpy as np

import fca
import fca.algorithms
import ptlm_wsid.target_context as tc

logger = logging.getLogger(__name__)


def fca_cluster(doc2preds):
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
        att_del = []
        att_th = max(3, np.log10(len(cxt.objects)))
        logger.debug(f'Att extent threshold: {att_th}')
        for att in cxt.attributes:
            if len(cxt.get_attribute_extent(att)) < att_th:
                att_del.append(att)
        cxt.delete_attributes(att_del)
        logger.debug(f'Number of attributes after cleaning: {len(cxt.attributes)}')
        cxt = cxt.clarify_objects()
        return cxt

    def get_sense(factor):
        docs = list(factor.extent)
        term_rank = dict()
        common_terms = factor.intent
        for term in common_terms:
            term_rank[term] = sum(doc2preds[doc].index(term) for doc in docs)
        sense_ids = sorted(term_rank, key=lambda x: term_rank[x])[:5]
        sense = fca.Concept(docs, set(sense_ids))
        return sense

    cxt = get_cxt()
    cxt = filter_cxt(cxt)
    chosen_factors = []
    chosen_scores = []
    chosen_senses = []
    factors_iter = fca.algorithms.factors.algorithm2_weighted(cxt, fidelity=0.8)
    try:
        for i, (factor, factor_score) in enumerate(factors_iter):
            cluster_threshold = sum(chosen_scores) / (len(chosen_scores) + 3) if chosen_scores else 0
            if factor_score < cluster_threshold:
                break
            n_unique_atts = len(factor.intent - {x for j in range(len(chosen_factors)) for x in chosen_factors[j].intent})
            n_unique_objects = len(factor.extent - {x for j in range(len(chosen_factors)) for x in chosen_factors[j].extent})
            overlap = [1 - (len(factor.intent - chosen_factors[j].intent) / len(factor.intent) *
                            len(factor.extent - chosen_factors[j].extent) / len(factor.extent))
                       for j in range(len(chosen_factors))]
            cluster_score = factor_score*(n_unique_objects / len(factor.extent) * n_unique_atts / len(factor.intent))

            logger.debug(f'### NEXT FACTOR ###')
            logger.debug(f'Factor # {i}, score: {factor_score}')
            logger.debug(
                f'Extent: {len(factor.extent)}, intent: {len(factor.intent)}')
            logger.debug(f'Extent: {factor.extent}')
            logger.debug(f'Unique intent: {n_unique_atts}')
            logger.debug(f'Unique extent: {n_unique_objects}')
            logger.debug(f'Cluster score = {cluster_score}')
            logger.debug(f'Threshold = {cluster_threshold}')
            logger.debug(f'Overlaps:' + '\t'.join(f'{i}: {overlap[i]}'
                                                  for i in range(len(overlap))))

            if n_unique_atts > n_unique_objects > 1 and \
                    cluster_score > cluster_threshold and \
                    max(overlap, default=0) < 0.5:
                new_sense = get_sense(factor)
                if any(new_sense.intent & s.intent for s in chosen_senses):
                    continue
                logger.debug('Senser added.')
                chosen_factors.append(factor)
                chosen_scores.append(cluster_score)
                chosen_senses.append(new_sense)
    except AssertionError:
        pass
    logger.info(f'Factors chosen: {len(chosen_factors)}')
    logger.info('\nFactor #'.join([f'{i}: ' + str(factor.extent) + '\n' + str(factor.intent)
                                   for i, factor in enumerate(chosen_factors)]))
    return chosen_senses


def induce(contexts, target_start_end_tuples, top_n_pred=100):
    predicted = dict()
    for i, (text_cxt, target_start_end) in enumerate(zip(contexts,
                                               target_start_end_tuples)):
        cxt = tc.TargetContext(text_cxt, target_start_end)
        top_pred = cxt.get_topn_predictions(top_n=top_n_pred)
        predicted[i] = top_pred
    senses = fca_cluster(predicted)
    return senses


if __name__ == '__main__':
    import context_embeddings as ce
    top_n_pred = 25
    for text, se in zip(ce.texts, ce.ses):
        tcxt = tc.TargetContext(text, se)
        top_pred = tcxt.get_topn_predictions(top_n=25)
        print(text)
        print(top_pred)