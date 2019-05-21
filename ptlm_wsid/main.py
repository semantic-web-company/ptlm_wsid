import logging

logging.basicConfig(
        # level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    )
import os
from collections import defaultdict, Counter
from pathlib import Path

from scipy.special import expit, softmax
from dotenv import load_dotenv
import torch
import numpy as np

import target_context as tc
import cluster

logger = logging.getLogger(__name__)

env_path = Path('.') / '.env'
load_dotenv(verbose=True, dotenv_path=env_path)


def get_corpus(entity_corpus_path):
    forms_path = os.path.join(entity_corpus_path, 'forms.tsv')
    with open(forms_path) as forms_f:
        forms = forms_f.read().lower().split('\t')
    corpus = []
    target_labels = forms
    for text_name in os.listdir(entity_corpus_path):
        if not text_name.endswith('txt'): continue
        text_path = os.path.join(entity_corpus_path, text_name)
        with open(text_path) as f:
            txt = f.read()
        corpus.append(tc.Document(text_name, txt))
    return corpus, target_labels


def main(data_path):
    for entity_title in os.listdir(data_path):
        entity_corpus_path = os.path.join(data_path, entity_title)
        if (entity_title.startswith('.') or
                not os.path.isdir(entity_corpus_path)):
            continue
        corpus, target_labels = get_corpus(entity_corpus_path)

        print(f'\n\n#### Corpus for {target_labels} loaded. {len(corpus)} docs.')

        predicted = defaultdict(list)
        top_n = 100
        for i_doc, doc in enumerate(corpus):
            target_cxts = doc.extract_target_cxts(target_labels=target_labels)
            for text_cxt, target_start_end in target_cxts:
                cxt = tc.SubstituteTargetContext(text_cxt, target_start_end)
                top_pred, top_logit_scores = cxt.get_topn_predictions(top_n=top_n)
                predicted[doc.name] += top_pred
            if len(predicted[doc.name]) > top_n:
                pred = [x[0] for x in Counter(predicted[doc.name]).most_common(n=top_n)]
                predicted[doc.name] = pred

        assert len(predicted) == len(corpus)
        assert all(len(preds) == top_n for preds in predicted.values()), [len(x) for x in predicted.values()]
        senses = cluster.fca_cluster(predicted)
        print(f'Factors chosen: {len(senses)}')
        print('\nFactor #'.join(
            [f'{i}: ' + str(factor.extent) + '\n' + str(factor.intent) for
             i, factor in enumerate(senses)]))

        sense_descriptors = [cl.intent for cl in senses]
        sense_similarities = tc.sense_similarities(sense_descriptors,
                                                   targets=['cocktail', 'drink', 'mix'])
        sense_embeddings = [tc.embed(s_descrs) for s_descrs in sense_descriptors]
        same_pred = 0
        for i_doc, doc in enumerate(corpus):
            # dis_r_doc = [0] * len(sense_descriptors)
            # target_cxts = doc.extract_target_cxts(target_labels=target_labels)
            # for text_cxt, target_start_end in target_cxts:
            #     cxt = tc.SubstituteTargetContext(text_cxt, target_start_end)
            #     dis_r_cxt = cxt.disambiguate(sense_clusters=sense_descriptors,
            #                                  top_n=100)
            #     for i in range(len(dis_r_doc)):
            #         dis_r_doc[i] += dis_r_cxt[i]

            pred_emb = tc.embed(predicted[doc.name])
            dis_r_doc_cos = [torch.cosine_similarity(pred_emb, sense_emb).
                             data.tolist()[0]
                     for sense_emb in sense_embeddings]
            print(doc.name)
            print(f'Cosine sim: {dis_r_doc_cos}')
            dis_r_doc_ol = tc.SubstituteTargetContext.self_count_overlaps(
                predicted[doc.name], sense_descriptors)
            print(f'Overlap sim: {dis_r_doc_ol}')
            print()
            if np.argmax(dis_r_doc_cos) == np.argmax(dis_r_doc_ol):
                same_pred += 1
        print(f'Same predictions: {same_pred}')


if __name__ == '__main__':
    cocktailpath = '/home/revenkoa/local_data/thesaural_wsi-master/cocktails'
    main(data_path=cocktailpath)
    # t2v_items, labels = cluster_external_scode('/home/revenkoa/Dropbox/personal/Projects/wsd/ptrnn_wsid/ptrnn_wsid/americano.tsv')
