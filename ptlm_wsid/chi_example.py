from collections import defaultdict, Counter
from io import StringIO

import fca
from ptlm_wsid.chi import parse_conll, collect_ners, get_senses_per_ne
from ptlm_wsid.utils import get_clean_cxt

if __name__ == '__main__':
    from urllib.request import urlopen

    # 1
    dataset_url = 'https://github.com/elenanereiss/Legal-Entity-Recognition/raw/master/data/ler.conll'
    t_limit = 1000
    with urlopen(dataset_url) as ler_file:
        ler_contents = ler_file.read().decode('utf-8')
    ler_lines_tabbed = '\n'.join(x.replace(' ', '\t') for x in ler_contents.split('\n'))
    ler_lines_tabbed = StringIO(ler_lines_tabbed)
    print(f'Data downloaded from {dataset_url}')
    data = parse_conll(ler_lines_tabbed, n_tokens=t_limit)
    all_forms, all_tags = list(zip(*list(data)))
    print(f'Total {len(all_forms)} tokens')
    # 2
    ners, tags, contexts, start_ends = collect_ners(all_forms, all_tags,
                                                    tokens_window=25)
    ner_agg = defaultdict(list)
    for i, (ner, tag) in enumerate(zip(ners, tags)):
        assert ner == contexts[i][start_ends[i][0]:start_ends[i][1]]
        ner_agg[f'{ner}::{tag}'].append(i)
    print(f'Total NE occurrences: {len(ners)}, Total unique NEs: {len(ner_agg)}')
    # 3
    ners_dict = get_senses_per_ne(ner_agg, contexts, start_ends, th_att_len=4)
    print(f'NE senses obtained, total: {len(ners_dict)}')
    # 4
    cxt = get_clean_cxt(ners_dict, min_att_extent=3)
    print(f'Context prepared. objects: {len(cxt.objects)}, '
          f'attributes: {len(cxt.attributes)}')
    # 5
    factors_iter = fca.algorithms.factors.algorithm2_w_condition(
        cxt, fidelity=0.85,
        allow_repeatitions=False,
        min_atts_and_objs=4
    )
    print('Factors:')
    for i, (factor, agg_score) in enumerate(factors_iter):
        factor_ne_cls = Counter([obj.split('::')[-1].split('##')[0]
                                 for obj in factor.extent])
        out = f'Factor {i}.\n'
        out += f'{", ".join(factor.intent)}.\n'
        out += f'Contained classes {factor_ne_cls}\n'
        out += f'Accumulated score: {agg_score:0.4f}'
        out += f'Total NEs: {len(factor.extent)}, total descriptors: {len(factor.intent)}'
        print(out)