import json
import os
from collections import defaultdict, Counter

from conllu import parse

from ptlm_wsid.target_context import TargetContext
from ptlm_wsid.generative_factors import induce, fca_cluster

# ler_path = os.getenv('LER_PATH', default='/home/revenkoa/local_data/LER/ler_tab.conll')


def read(ler_path=os.getenv('LER_PATH', default='/home/revenkoa/local_data/datasets/LER/dfki/ler_tab.conll'),
         read_limit=-1):
    with open(ler_path) as ler_file:
        data = parse(''.join(ler_file.readlines(read_limit)),
                     fields=['form', 'tag'],
                     field_parsers={'tag': lambda line, i: line[i].split('-')})
    return data


def collect_ners(read_limit=500000, do_print_stats=True):
    def print_stats(ners_dict):
        s = 0
        ks = 0
        for k, v in ners_dict.items():
            if len(v) > 1:
                s += len(v)
                ks += 1
        stats = f'Total unique entities: {len(ners_dict)}, ' \
                f'total NE occurrences: {sum(len(x) for x in ners_dict.values())}, ' \
                f'entities in multiple sents: {ks}, ' \
                f'total sents with repeating entities: {s}, total sents: {len(ler_data)}'
        print(stats)

    def _save_ner(ner_form, ner_tag, sent, index):
        ner_str = ' '.join(ner_form)
        start_ind = sum(len(token['form']) for token in sent[:index]) + index
        # print(start_ind, index, ner_form, sent)
        # assert start_ind == cxt.index(ner_str), (cxt, start_ind, ner_str, cxt.index(ner_str), cxt[start_ind-5:start_ind+5], index)
        end_ind = start_ind + len(ner_str)
        tc = TargetContext(
            context=cxt,
            target_start_end_inds=(start_ind, end_ind))
        assert tc.target_str == ner_str
        ners_dict[ner_str].append(tc)
        tags_dict[ner_str].append(ner_tag)
        assert len(ners_dict) == len(tags_dict), (len(ners_dict), len(tags_dict))

    ler_data = read(read_limit=read_limit)

    ners_dict = defaultdict(list)
    tags_dict = defaultdict(list)
    for sent in ler_data:
        ner_form = None
        cxt = ' '.join(token['form'] for token in sent)
        for index, token in enumerate(sent):
            if token['tag'][0] == 'B':
                ner_form = [token['form']]
                ner_begin_ls_index = index
                ner_tag = token['tag'][1]
            elif token['tag'][0] == 'I':
                ner_form.append(token['form'])
            elif token['tag'][0] == 'O':
                if ner_form is not None:
                    _save_ner(ner_form, ner_tag, sent, ner_begin_ls_index)
                    ner_form = None
            else:
                assert 0, token
        if ner_form is not None:
            _save_ner(ner_form, ner_tag, sent, ner_begin_ls_index)

    if do_print_stats:
        print_stats(ners_dict)
    return ners_dict, tags_dict


def main(len_contexts_per_ne_limit=10,
         autosave_freq=250):
    """

    :param autosave_freq: how often results are saved, i.e. after every X NEs
    :param len_contexts_per_ne_limit: if an NE is presented in less contexts it is skipped
    :return:
    """
    ners, tags = collect_ners(read_limit=-1)
    ner2preds = dict()
    cnt = 0
    for ner_form, tcs in ners.items():
        if len(tcs) > len_contexts_per_ne_limit:
            ner_tags = Counter(tags[ner_form])
            ner_tag = ner_tags.most_common(1)[0][0]
            if len(tcs) > 15:
                tcs = tcs[:15]
            cnt += 1
            print(
                f'NE #{cnt} of {len(ners)}. '
                f'Surface form: {ner_form}, '
                f'NE tags: {ner_tags}, '
                f'# contexts: {len(tcs)}')
            senses = induce([tc.context for tc in tcs],
                            [tc.target_start_end_inds for tc in tcs],
                            n_sense_indicators=10,
                            target_pos='N',
                            # do_mask=False,
                            lang='deu')
            if senses:
                ner2preds[f'({ner_form}, {ner_tag})'] = list(sum((list(s.intent) for s in senses), []))
            print(f'# senses induced: {len(senses)}')
            for i, sense in enumerate(senses):
                print(f'Sense descriptors: {sense.intent}, '
                      f'found in contexts {sense.extent}')
            #
            if cnt % autosave_freq == 0:
                print(f'\nNumber of processed NER is {len(ner2preds)}')
                with open(f'ners{len(ner2preds)}.json', 'w') as f:
                    json.dump(ner2preds, f)
    # final result
    print(f'\nNumber of processed NER is {len(ner2preds)}')
    with open(f'ners{len(ner2preds)}.json', 'w') as f:
        json.dump(ner2preds, f)


if __name__ == '__main__':
    # main(len_contexts_per_ne_limit=3)
    ners, tags = collect_ners(read_limit=-1)
    with open('./all_ners.tsv', 'w') as f:
        for ner, tag in tags.items():
            f.write(f'{ner}\t{tag}\n')