import os
from collections import Counter
from pathlib import Path
import numpy as np

from tqdm import trange

from ptlm_wsid.target_context import TargetContext


def estimate_thresholds(contexts, target_inds, hypernyms, definitions, labels):
    hyp_scores = []
    def_scores = []
    acc_scores = []
    for i in trange(len(contexts)):
        hyp_score, def_score = score(contexts[i], target_inds[i],
                                     hypernyms[i], definitions[i])
        combined_score = (hyp_score + def_score) / 2
        hyp_scores.append(hyp_score)
        def_scores.append(def_score)
        acc_scores.append(combined_score)
    p_ratio = sum(labels) / len(labels)
    all_scores = [hyp_scores, def_scores, acc_scores]
    acc1 = dict()
    acc2 = dict()
    acc3 = dict()
    for th_value_int in range(35, 160):
        th_value = th_value_int / 100
        acc = [compute_accuracy(scores, th_value, labels) for scores in all_scores]
        acc1[acc[0]] = th_value
        acc2[acc[1]] = th_value
        acc3[acc[2]] = th_value
    acc_maxs = [max(acc1.keys()), max(acc2.keys()), max(acc3.keys())]
    ths = [acc1[acc_maxs[0]],
           acc2[acc_maxs[1]],
           acc3[acc_maxs[2]]]
    print(acc_maxs)
    print(ths)
    return ths


def compute_accuracy(scores, th, labels):
    preds = [x > th for x in scores]
    correct = sum([p == l for p, l in zip(preds, labels)])
    acc_score = correct / len(scores)
    return acc_score


def read_wic_tsv_ds(folder_path: Path):
    contexts = []
    target_inds = []
    examples_path = next(folder_path.glob('*_examples.txt'))
    with examples_path.open() as ex_f:
        ex_f_lines = ex_f.readlines()
        for line in ex_f_lines:
            _, target_ind, context = line.split('\t')
            target_ind = int(target_ind.strip())
            clean_cxt = context.strip()
            contexts.append(clean_cxt)
            target_inds.append(target_ind)
    hypernyms_path = next(folder_path.glob('*_hypernyms.txt'))
    with hypernyms_path.open() as hyp_f:
        hypernyms = [[hyp.replace('_', ' ').strip() for hyp in line.split('\t')]
                     for line in hyp_f.readlines()]
    defs_path = next(folder_path.glob('*_definitions.txt'))
    with defs_path.open() as defs_f:
        definitions = defs_f.readlines()
    try:
        labels_path = next(folder_path.glob('*_labels.txt'))
        with labels_path.open() as labels_f:
            labels = [x.strip() == 'T' for x in labels_f.readlines()]
    except:
        labels = None
    assert len(contexts) == len(hypernyms) == len(definitions), \
        (len(contexts), len(hypernyms), len(definitions))
    return contexts, target_inds, hypernyms, definitions, labels


def score(context, target_ind, hypernyms, definition):
    target = context.split(' ')[target_ind]
    t_start = context.find(target)
    t_end = t_start + len(target)
    tc = TargetContext(context=context, target_start_end_inds=(t_start, t_end))
    hyp_score, def_score = tc.disambiguate(
        definitions=[definition],
        sense_clusters=[hypernyms])[0]
    return hyp_score, def_score


def get_scores(contexts, target_inds, hypernyms, definitions, labels,
                     ths=(0.5, 0.5, 0.5)):
    hyp_scores = []
    def_scores = []
    combined = []
    for i in trange(len(contexts)):
        hyp_score, def_score = score(contexts[i], target_inds[i],
                                     hypernyms[i], definitions[i])
        combined_score = (hyp_score + def_score) / 2
        hyp_scores.append(hyp_score)
        def_scores.append(def_score)
        combined.append(combined_score)
    return hyp_scores, def_scores, combined


def predict(scores, th):
    preds = [x > th for x in scores]
    return preds
    # acc = compute_accuracy(hyp_scores, ths[0], labels)
    # print(f'\nhypernyms\nThreshold = {ths[0]}\n'
    #       f'Accuracy:{acc}, Average score: {sum(hyp_scores) / len(hyp_scores)}')
    # acc = compute_accuracy(def_scores, ths[1], labels)
    # print(f'\nDefinition\nThreshold = {ths[1]}\n'
    #       f'Accuracy:{acc}, Average score: {sum(def_scores) / len(def_scores)}')
    # acc = compute_accuracy(combined, ths[2], labels)
    # print(f'\nCombined\nThreshold = {ths[2]}\n'
    #       f'Accuracy:{acc}, Average score: {sum(combined) / len(combined)}')


if __name__ == '__main__':
    wic_tsv_dev = Path(os.getenv('WIC_TSV_DEV_PATH'))
    wic_tsv_test = Path(os.getenv('WIC_TSV_TEST_PATH'))
    error_msg = 'Please, set environment variables to point to train and test folders'
    assert wic_tsv_dev is not None, error_msg
    assert wic_tsv_test is not None, error_msg
    data = read_wic_tsv_ds(wic_tsv_dev)
    # contexts, target_inds, hypernyms, definitions, labels = data
    ths = estimate_thresholds(*data)

    a = read_wic_tsv_ds(wic_tsv_test)
    hyp_scores, def_scores, comb_scores = get_scores(*a, ths=ths)
    hyp_preds = predict(hyp_scores, ths[0])
    with open('./hyp_preds.out', 'w') as f:
        f.write('\n'.join(map(str, hyp_preds)))
    def_preds = predict(hyp_scores, ths[1])
    with open('./def_preds.out', 'w') as f:
        f.write('\n'.join(map(str, def_preds)))
    comb_preds = predict(hyp_scores, ths[2])
    with open('./comb_preds.out', 'w') as f:
        f.write('\n'.join(map(str, comb_preds)))

"""Results:
Predictions: def_preds_distilbert.out
{'acc': 0.5689127105666156, 'F_1': 0.3415204678362573, 'P': 0.7604166666666666, 'R': 0.22021116138763197}

Predictions: comb_preds_distilbert.out
{'acc': 0.611791730474732, 'F_1': 0.5129682997118156, 'P': 0.7063492063492064, 'R': 0.40271493212669685}

Predictions: hyp_preds_distilbert.out
{'acc': 0.622511485451761, 'F_1': 0.6020984665052461, 'P': 0.6475694444444444, 'R': 0.5625942684766214}

Predictions: def_preds_bert.out
{'acc': 0.5436447166921899, 'F_1': 0.26237623762376233, 'P': 0.7310344827586207, 'R': 0.15987933634992457}

Predictions: comb_preds_bert.out
{'acc': 0.6049004594180705, 'F_1': 0.5186567164179104, 'P': 0.6797066014669927, 'R': 0.4193061840120664}

Predictions: hyp_preds_bert.out
{'acc': 0.6278713629402757, 'F_1': 0.6009852216748769, 'P': 0.6594594594594595, 'R': 0.5520361990950227}
"""
