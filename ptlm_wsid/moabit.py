import json
import os

import csv
import time
from collections import defaultdict, Counter
from pathlib import Path

import fca

import ptlm_wsid.ler as ler
import ptlm_wsid.generative_factors as gf
import ptlm_wsid.target_context as tc


O_tags = ['"', 'NE', 'O']


def read(ner_path=Path(os.getenv('NER_PATH', default='/home/revenkoa/local_data/datasets/NER/moabit_german')),
         read_limit=-1):
    data = []
    for filepath in ner_path.iterdir():
        if not str(filepath).endswith('.csv'): continue
        with filepath.open():
            file_text = filepath.read_text()
        lines = file_text.split('\n')[1:]  # skip the first line - column names
        file_data = []
        for line in lines:
            file_data.append(line.split(';'))
        data.append(file_data)
    print(f'Total files processed: {len(data)}')
    print(f'total number of tokens: {sum(len(x) for x in data)}')
    return data


def collect_ners(forms, tags, tokens_window=35):
    assert len(forms) == len(tags)
    ners = []
    contexts = []
    ner_tags = []
    start_ends = []

    prev_tag = None
    ner_form = []
    for i, (form, tag) in enumerate(zip(forms, tags)):
        # ner_form = None
        # cxt = ' '.join(token['form'] for token in sent)
        if tag not in O_tags and (prev_tag in O_tags + [None] or
                                  prev_tag == tag):
            if form.strip():  # no empty string
                ner_form.append(form)
                prev_tag = tag
        elif tag in O_tags or prev_tag != tag:
            if ner_form:
                ners.append(' '.join(ner_form))
                ner_tags.append(prev_tag)
                prev_tag = tag
                cxt_before = ' '.join(forms[(i - tokens_window if i > tokens_window else 0):i])
                start_ind = len(cxt_before)
                end_ind = start_ind + len(ner_form) + 1
                ner_form = []
                cxt = ' '.join(forms[(i - tokens_window if i > tokens_window else 0):
                                     (i + tokens_window if i + tokens_window < len(forms)
                                      else None)])
                contexts.append(cxt)
                start_ends.append( (start_ind, end_ind) )

    return ners, ner_tags, contexts, start_ends


def get_forms_tags(data):
    forms = []
    tags = []
    for doc in data:
        for line in doc:
            try:
                form, tag1, tag2 = line
                forms.append(form)
                tags.append(tag1)
                # out.append((form, tag2 if tag2 else tag1))
            except ValueError as e:
                # print(line)
                # print(e)
                pass
    return forms, tags


def induce(contexts, target_start_end_tuples, top_n_pred=100,
           titles=None, n_sense_indicators=5, target_pos=None, lang='eng',
           do_mask=True, min_number_contexts=-1):
    predicted = dict()
    senses = []
    if len(contexts) > min_number_contexts:
        for i, (text_cxt, target_start_end) in enumerate(zip(contexts,
                                                   target_start_end_tuples)):
            cxt = tc.TargetContext(text_cxt, target_start_end)

            start_t = time.time()
            top_pred = cxt.get_topn_predictions(top_n=top_n_pred,
                                                target_pos=target_pos,
                                                lang=lang,
                                                do_mask=do_mask)
            predicted[titles[i] if titles else i] = top_pred
            print(f'Predictions took {time.time()-start_t:0.3f}')
            if len(contexts) > 3:  # less does not make sense
                senses = gf.fca_cluster(predicted, n_sense_indicators=n_sense_indicators)
        if not senses:
            all_predicted = sum(predicted.values(), [])
            top_predicted = [x[0] for x in Counter(all_predicted).most_common(
                n_sense_indicators)]
            senses = [fca.Concept(intent=top_predicted,
                                  extent=list(predicted.keys()))]
    return senses


def get_ner_substitutes(
        ners, tags, cxts, start_ends,
        len_contexts_per_ne_limit=10,
        n_sense_descriptors=10,
        top_n_pred=100,
        skip_first_ners=None,
        yield_freq=250):
    """

    :param yield_freq: how often results are saved, i.e. after every X NEs
    :param len_contexts_per_ne_limit: if an NE is presented in less contexts it is skipped
    :return:
    """
    ner2preds = dict()

    ner2inds = {
        ner: [i for i, x in enumerate(ners) if x == ner] for ner in set(ners)}
    for j, (ner_form, ner_indices) in enumerate(ner2inds.items()):
        if skip_first_ners is not None and j <= skip_first_ners:
            continue
        if not len({tags[i] for i in ner_indices}) == 1:
            print(ner_form, {tags[i] for i in ner_indices})
        ner_tag = tags[ner_indices[0]]
        ner_cxts = [cxts[i] for i in ner_indices]
        ner_ses = [start_ends[i] for i in ner_indices]
        print(
            f'NE #{j} of {len(ner2inds)}. '
            f'Surface form: {ner_form}, '
            f'NE tags: {ner_tag}, '
            f'# contexts: {len(ner_cxts)}')
        if len(ner_cxts) > 15:
            ner_cxts = ner_cxts[:15]
            ner_ses = ner_ses[:15]
        senses = induce(ner_cxts,
                        ner_ses,
                        n_sense_indicators=n_sense_descriptors,
                        top_n_pred=top_n_pred,
                        target_pos='N',
                        # do_mask=False,
                        lang='deu')
        print(f'# senses induced: {len(senses)}')
        print(f'Contexts: {ner_cxts}')
        if senses:
            for sense in senses:
                print(f'Sense descriptors: {sense.intent}, '
                      f'found in contexts {sense.extent}')
            #
            all_sense_identifiers = set()
            for sense in senses:
                all_sense_identifiers |= sense.intent
            ner2preds[', '.join([ner_form, ner_tag])] = list(all_sense_identifiers)
        if j % yield_freq == 0:
            print(f'\nNumber of processed NER is {len(ner2preds)}')
            yield ner2preds
    # final result
    print(f'\nNumber of processed NER is {len(ner2preds)}')
    yield ner2preds


if __name__ == '__main__':
    import logging
    # logging.basicConfig(level=logging.DEBUG)

    data = read()
    forms, tags = get_forms_tags(data)
    # for form, tag in zip(forms, tags):
    #     if tag == '"' or tag == 'NE':
    #         print(form, tag)
    ners, tags, cxts, start_ends = collect_ners(forms, tags)
    print(len(ners), len(forms), len(tags))
    print(set(tags))
    # all_ners_file = Path('all_ners.tsv')
    # with all_ners_file.open('w') as f:
    #     for (form, cls), freq in Counter(zip(ners, tags)).most_common():
    #         f.write('\t'.join([form, cls, str(freq)]))
    #         f.write('\n')
    print(Counter(zip(ners, tags)).most_common(100))
    print(Counter(Counter(ners).values()))
    for ners_dict in get_ner_substitutes(ners, tags, cxts, start_ends,
                                         yield_freq=250,
                                         n_sense_descriptors=25,
                                         top_n_pred=50,
                                         # skip_first_ners=250
                                         ):
        with open(f'ners{len(ners_dict)}.json', 'w') as f:
            json.dump(ners_dict, f)
