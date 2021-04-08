from collections import defaultdict, Counter
from io import StringIO
from typing import Iterator, List, Iterable, Tuple

import conllu

import fca
from ptlm_wsid.chi import collect_ners, iter_senses
from ptlm_wsid.utils import get_cxt, clean_cxt


def parse_conll(data_str: str, fields=('form', '1', '2', 'tag'),
                n_tokens: int = -1) -> Iterator[Tuple[str, str]]:
    data = conllu.parse(data_str, fields=fields,
                        field_parsers={'tag': lambda line, i: line[i].split('-')})
    i = 0
    for sent in data:
        for w in sent:
            i += 1
            if 0 < n_tokens < i:
                return
            yield w['form'], w['tag']


def iter_ners_dict(dataset: str,
                   fields: Iterable[str] = ('form', 'tag'),
                   t_limit=120000,
                   n_sense_descriptors=25,
                   th_att_len=4,
                   lang='deu') -> Iterator[Tuple[str, List[str]]]:
    data = parse_conll(dataset, fields=fields, n_tokens=t_limit)
    all_forms, all_tags = list(zip(*list(data)))
    print(f'Total {len(all_forms)} tokens')
    ners, tags, contexts, start_ends = collect_ners(all_forms, all_tags,
                                                    tokens_window=25)
    ner_agg = defaultdict(list)
    for i, (ner, tag) in enumerate(zip(ners, tags)):
        assert ner == contexts[i][start_ends[i][0]:start_ends[i][1]]
        ner_agg[f'{ner}::{tag}'].append(i)
    senses_iter = iter_senses(ner_agg, contexts, start_ends,
                              lang=lang, cxts_limit=50, n_pred=50,
                              n_sense_descriptors=n_sense_descriptors,
                              target_pos='N', th_att_len=th_att_len)
    for ner_form, ner_senses in senses_iter:
        yield ner_form, ner_senses


if __name__ == '__main__':
    from urllib.request import urlopen
    dataset_url = 'https://github.com/elenanereiss/Legal-Entity-Recognition/raw/master/data/ler.conll'
    t_limit = 1000
    with urlopen(dataset_url) as ler_file:
        ler_contents = ler_file.read().decode('utf-8')
    ler_tabbed = '\n'.join(x.replace(' ', '\t') for x in ler_contents.split('\n'))
    ners_dict = {f'{ner_form}##{i}': list(sense)
                 for ner_form, senses in iter_ners_dict(dataset=ler_tabbed,
                                                        t_limit=t_limit)
                 for i, sense in enumerate(senses)}
    print(f'NE senses obtained, total: {len(ners_dict)}')
    cxt = get_cxt(ners_dict)
    print(f'Binary matrix prepared. NEs: {len(cxt.objects)}, '
          f'descriptors: {len(cxt.attributes)}')
    cxt = clean_cxt(cxt, min_att_extent=3)
    print(f'Binary matrix cleaned. '
          f'All descriptors with less than '
          f'3 corresponding entities removed for speedup. '
          f'descriptors: {len(cxt.attributes)}, '
          f'NEs: {len(cxt.objects)}, ')
    factors_iter = fca.algorithms.factors.algorithm2_w_condition(
        cxt, fidelity=0.8,
        allow_repeatitions=False,
        min_atts_and_objs=4
    )
    print('New classes:')
    for i, (cls, cls_score, agg_score) in enumerate(factors_iter):
        factor_ne_cls = Counter([obj.split('::')[-1].split('##')[0]
                                 for obj in cls.extent])
        out = f'Class {i}.\n'
        out += f'{", ".join(cls.intent)}.\n'
        out += f'Contained NE types {factor_ne_cls}\n'
        out += f'Class score: {cls_score:0.4f}, accumulated score: {agg_score:0.4f}\n'
        out += f'Total NEs: {len(cls.extent)}, total descriptors: {len(cls.intent)}'
        print(out)