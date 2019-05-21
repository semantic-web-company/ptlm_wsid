import logging
import os
import subprocess

from sklearn.cluster import KMeans
import numpy as np

import fca
import fca.algorithms

logger = logging.getLogger(__name__)
outfolder_path = os.getenv('SCODE_FOLDER', default='/tmp/')


def cluster_external_scode(doc2preds, target_label):
    def write_scode_file(doc2preds, target_label):
        outfile = f'{target_label}.tsv'
        outfile_path = os.path.join(outfolder_path, outfile)

        if os.path.exists(outfile_path):
            os.remove(outfile_path)
        with open(outfile_path, 'a+') as outf:
            for i_doc, (docname, preds) in enumerate(doc2preds.items()):
                for pred_token in preds:
                    outf.write(f'{target_label}{i_doc}\t{pred_token}\n')

        return outfile_path

    def best_k(X):
        k2loss = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            k2loss.append((k, kmeans.inertia_))
        k_ind_score = (0, 0)
        for i, (k, loss) in enumerate(k2loss):
            if i < 1 or i == len(k2loss) - 1: continue
            loss_1, loss1 = k2loss[i - 1][1], k2loss[i + 1][1]
            score = (loss_1 - loss) - (loss - loss1)
            if score > k_ind_score[1]:
                k_ind_score = (k, score)
            print(k, score)
        k = k_ind_score[0]
        return k

    def print_clusters(tokens, labels):
        t2l = dict(zip(tokens, labels))
        set_l = set(labels)
        cluster_str = ''
        for cluster in set_l:
            cluster_str += f'Cluster #{cluster}:\n'
            for token, label in t2l.items():
                if label == cluster:
                    cluster_str += f'{token}\t'
            cluster_str += '\n'
        print(cluster_str)

    file_path = write_scode_file(doc2preds, target_label)
    out = subprocess.check_output(f"scode < {file_path}", shell=True)
    out = out.decode("utf-8")
    lines = [line.split('\t') for line in out.split('\n') if line]
    token2vec = {line[0]: [float(el) for el in line[2:]]
                 for line in lines if line}
    t2v_items = list(token2vec.items())
    X = np.array([x[1] for x in t2v_items])
    k = best_k(X)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    return t2v_items, labels


def fca_cluster(doc2preds, objects_impact_factor=0.5):
    def get_cxt():
        # Prepare the formal context
        intents = []
        i2docname = dict()
        objs = []
        for i, (docname, preds) in enumerate(doc2preds.items()):
            preds = set(preds)
            intents.append(preds)
            objs.append(docname)
            i2docname[i] = docname
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
        return cxt

    def add_factor(factors_ls, new_factor):
        factors_ls.append(new_factor)
        logger.debug('Factor added!')

    def enrich_factor(chosen_factors, new_factor):
        if chosen_factors:
            new_factor.extent = set()
            for obj_i, obj_intent in enumerate(cxt.intents()):
                object_ = cxt.objects[obj_i]
                overlap_new = len(new_factor.intent & obj_intent) / len(
                    new_factor.intent)
                overlap_other = max(len(factor.intent & obj_intent) /
                                    len(factor.intent)
                                    for factor in chosen_factors)
                if overlap_new > overlap_other:
                    new_factor.extent.add(object_)

    cxt = get_cxt()
    cxt = filter_cxt(cxt)
    # Choose the cluster threshold
    av_att_len = np.mean([len(x) for x in cxt.intents()])
    cluster_score_threshold = np.log(len(cxt.objects))/len(cxt.objects) * (1 - 1/np.log(av_att_len))
    # Iterate over factors
    factors_iter = fca.algorithms.factors.algorithm2(cxt, fidelity=0.8)
    chosen_factors = []
    for i, (factor, factor_score) in enumerate(factors_iter):
        if factor_score <= 0.007: break  # criterium to break the iteration: just a speed up
        enrich_factor(chosen_factors, factor)
        n_unique_atts = len(factor.intent - {x for j in range(len(chosen_factors)) for x in chosen_factors[j].intent})
        n_unique_objects = len(factor.extent)  # len(factor.extent - {x for j in range(len(chosen_factors)) for x in chosen_factors[j].extent})
        logger.debug(f'### NEXT FACTOR ###')
        logger.debug(f'Factor # {i}, score: {factor_score}')
        logger.debug(f'Extent: {len(factor.extent)}, intent: {len(factor.intent)}')
        logger.debug(f'Extent: {factor.extent}')
        logger.debug(f'Unique intent: {n_unique_atts}')
        logger.debug(f'Unique extent: {n_unique_objects}')
        # if n_unique_objects < 2 or n_unique_atts < n_unique_objects:
        #     continue

        overlap = [1 - (len(factor.intent - chosen_factors[j].intent) /
                        len(factor.intent))
                   for j in range(len(chosen_factors))]
        logger.debug(f'Overlaps:' + '\t'.join(f'{i}: {overlap[i]}' for i in range(len(overlap))))
        # unique_objs_score = 1 - objects_impact_factor / n_unique_objects
        if chosen_factors:
            # cluster_score = (n_unique_objects / len(factor.extent) * n_unique_atts / len(factor.intent))*unique_objs_score - max(overlap)
            cluster_score = (n_unique_objects / len(cxt.objects) * n_unique_atts / len(factor.intent)) - max(overlap)
            logger.debug(f'Cluster score = {cluster_score}')
            if cluster_score > cluster_score_threshold:
                add_factor(chosen_factors, factor)
        if not chosen_factors:
            # cluster_score_threshold *= unique_objs_score
            logger.debug(f'threshold = {cluster_score_threshold}, average intent length: {av_att_len}')
            add_factor(chosen_factors, factor)

    logger.info(f'Factors chosen: {len(chosen_factors)}')
    logger.info('\nFactor #'.join([f'{i}: ' + str(factor.extent) + '\n' + str(factor.intent) for i, factor in enumerate(chosen_factors)]))
    return chosen_factors