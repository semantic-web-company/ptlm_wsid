import json
import logging, logging.config
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import configparser

import conllu

import fca
from ptlm_wsid.chi import collect_ners, iter_senses
from linking.dummy_linker import DummyLinker
from linking.wikidata_linker import WikidataLinker
from ptlm_wsid.utils import get_cxt, clean_cxt
# from scripts.chi_example import iter_ners_dict


# def get_ners_dict(dataset: str,
#                   fields: Iterable[str],
#                   ners_pred_target_folder: Path,
#                   n_sense_descriptors=25,
#                   th_att_len=4,
#                   t_limit=120000,
#                   lang='deu'):
#     ners_dict = dict()
#     senses_iter = iter_ners_dict(dataset=dataset, fields=fields,
#                                  t_limit=t_limit, lang=lang,
#                                  th_att_len=th_att_len,
#                                  n_sense_descriptors=n_sense_descriptors)
#     for j, (ner_form, ner_senses) in enumerate(senses_iter):
#         if len(ner_senses) > 1:
#             for i, sense in enumerate(ner_senses):
#                 ners_dict[f'{ner_form}##{i}'] = list(sense)
#         else:
#             ners_dict[f'{ner_form}'] = list(ner_senses[0])
#         if j % 250 == 0:
#             with (ners_pred_target_folder / f'preds{j}.json').open('w') as f:
#                 json.dump(ners_dict, f, indent=2)
#     with (ners_pred_target_folder / f'preds{j}.json').open('w') as f:
#         json.dump(ners_dict, f, indent=2)
#     return ners_dict

#
# def iter_factors(cxt, fidelity=0.5, min_atts_and_objs=4, allow_repeatitions=False):
#     params_dict = dict(fidelity=fidelity,
#                        allow_repeatitions=allow_repeatitions,
#                        min_atts_and_objs=min_atts_and_objs,
#                        objs_ge_atts=True)
#     factors_iter = fca.algorithms.factors.algorithm2_w_condition(
#         cxt, **params_dict)
#     print(f'fidelity: stops when this fraction of crosses in the table are covered. check aggregated score with each factor.')
#     print(f'Parameters: {params_dict}')
#     ###
#     print('New classes:')
#     for i, (factor, cls_score, agg_score) in enumerate(factors_iter):
#         factor_ne_cls = Counter([obj.split('::')[-1].split('##')[0]
#                                  for obj in factor.extent])
#         out = f'\nClass {i}.\n'
#         out += f'{", ".join(factor.intent)}.\n'
#         out += f'Contained NE types {factor_ne_cls}\n'
#         out += f'Class score: {cls_score:0.4f}, aggregated score: {agg_score:0.4f}\n'
#         out += f'Total NEs: {len(factor.extent)}, total descriptors: {len(factor.intent)}'
#         print(out)
#         out_json = {
#             'score': cls_score,
#             'descriptors': list(factor.intent),
#             'entities': list(factor.extent),
#             'types': factor_ne_cls
#         }
#         yield out_json


# def read_json_classes(folder: Path):
#     out = []
#     filenames = []
#
#     for filename in folder.glob(r'class*.json'):
#         with filename.open() as f:
#             contents = json.load(f)
#         out.append([contents['descriptors'], contents['entities']])
#         filenames.append(filename.name)
#     return out, filenames
#
#
# def read_factors(folder: Path):
#     descrs_pattern = re.compile(r'Descriptors:\n(([^\n]+\n)*)')
#     entities_pattern = re.compile(r'Entities:\n(([^\n]+\n?)*)')
#     out = []
#     filenames = []
#
#     for filename in folder.glob(r'class*.txt'):
#         with filename.open() as f:
#             contents = f.read()
#         m = re.search(descrs_pattern, contents)
#         descriptors = m.group(1).strip().split('\n')
#         m = re.search(entities_pattern, contents)
#         entities = m.group(1).strip().split('\n')
#         filenames.append(filename.name)
#         out.append((descriptors, entities))
#     return out, filenames
#
#
# def aggressive(cxt, chosen_classes, outpath='aggressive.txt'):
#     print('#' * 50)
#     print('Aggressive retagging of entities into selected classes')
#     print('#' * 50)
#     class2objs = defaultdict(list)
#     cls_objects = [cxt.aprime(cls) for cls in chosen_classes]
#     cls_atts_unions = [Counter(x for obj in cls_extent
#                                for x in cxt.get_object_intent(obj))
#                        for cls_extent in cls_objects]
#     for obj in cxt.objects:
#         obj_intent = cxt.get_object_intent(obj)
#         obj_scores = []
#         for cls_atts_union in cls_atts_unions:
#             score = (sum(cls_atts_union[a] for a in obj_intent) /
#                      sum(cls_atts_union.values()))
#             obj_scores.append(score)
#         obj_sense = chosen_classes[obj_scores.index(max(obj_scores))]
#         class2objs[', '.join(obj_sense)].append(obj)
#     with open(outpath, 'w') as f:
#         for chosen in chosen_classes:
#             chosen_str = ', '.join(chosen)
#             chosen_objs = class2objs[chosen_str]
#             intent_clss = Counter(obj.split('::')[-1].split('##')[0]
#                                   for obj in chosen_objs)
#             out = f'\n\nClass descriptors: {chosen_str}\n'
#             out += f'Total {len(chosen_objs)} entities acquired, their types:'
#             out += f'{intent_clss}\nEntities:\n'
#             out += ', '.join(chosen_objs)
#             f.write(out)
#
#
# def cautious(cxt, chosen_classes, outpath='cautious.txt'):
#     print('#' * 50)
#     print('Cautious retagging of entities into selected classes')
#     print('#' * 50)
#     out = ''
#     for intent in chosen_classes:
#         atts_extents = [cxt.get_attribute_extent(att) for att in intent]
#         obj_cnt = Counter(el for x in atts_extents for el in x)
#         att_cnt = Counter(obj_cnt.values())
#         th = round(max(len(att_cnt) / 2, len(att_cnt)*0.7))
#         intent_objs = [obj for obj, n_atts in obj_cnt.items() if n_atts >= th]
#         intent_clss = Counter(obj.split('::')[-1].split('##')[0]
#                               for obj in intent_objs)
#
#         out += (f'\n\nClass descriptors: {intent}\n' +
#                 ', '.join(f"{n_objs} objects have {n_atts} descriptors"
#                           for n_atts, n_objs in att_cnt.items()) +
#                 f'\nWe take only entities with at least {th} descriptors\n' +
#                 f'Total {len(intent_objs)} entities acquired, their types:' +
#                 f'{intent_clss}\nEntities:\n')
#         out += ', '.join(intent_objs)
#     with open(outpath, 'w') as f:
#         f.write(out)


def parse_conll_file(data_path: str, fields=('form', 'PoS', '2', 'tag'),
                     n_tokens: int = -1) -> Iterator[Tuple[str, str]]:
    with open(data_path) as f:
        data = conllu.parse_incr(f, fields=fields,
                                 field_parsers={'tag': lambda line, i: line[i].split('-')})
        i = 0
        for sent in data:
            for w in sent:
                i += 1
                if 0 < n_tokens < i:
                    return
                yield w['form'], w['tag']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Type induction on WikiNer corpus.')
    parser.add_argument('config_path',
                        # metavar='N', nargs='+',
                        type=str,
                        help='Relative (to wikiner.py script) path to the config file')
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    config_paths = [this_dir / args.config_path, this_dir / 'configs/logging.conf']
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_paths)
    logger = logging.getLogger()
    logging.config.fileConfig(config)
    logger.info(f'Config: {config.items("wikiner")}')
    # config into vars
    ners_predictions_output_stem = Path(config['wikiner']['ners_predictions_output_stem'])
    nes_in_cxts_path = Path(config['wikiner']['ner_contexts_output'])
    language = config['wikiner']['language']
    t_limit = int(config['wikiner']['tokens_to_be_processed'])
    conll_data_path = config['wikiner']['conll_data_path']
    ms = [int(m) for m in config['wikiner']['ms']]
    max_k = int(config['wikiner']['max_k'])
    th_n_descriptors = int(config['wikiner']['th_type_descriptors'])
    # Read data
    if nes_in_cxts_path.exists():
        logger.debug(f'Found {nes_in_cxts_path}, loading it.')
        # load from an existing file
        with open(nes_in_cxts_path) as f:
            f.readline()  # skip the first line - headers
            data_lines = f.readlines()
        nes_tags, start_inds, end_inds, contexts, orig_uris = tuple(zip(*[line.strip().split('\t') for line in data_lines if line.strip()]))
        start_inds = tuple(map(int, start_inds))
        end_inds = tuple(map(int, end_inds))
        nes, tags = tuple(zip(*[x.split('::') for x in nes_tags]))
        uris = []
        for i, uri in enumerate(orig_uris):
            if 'no.entity.found' in uri:
                uris.append(nes[i])
            else:
                uris.append(uri)
    else:
        logger.debug(f'No existing {nes_in_cxts_path} found.')
        data = parse_conll_file(conll_data_path, n_tokens=t_limit)
        logger.debug('Conll iter created')
        all_forms, all_tags = tuple(zip(*list(data)))
        nes, tags, contexts, start_ends = collect_ners(all_forms, all_tags, tokens_window=25)
        start_inds, end_inds = tuple(zip(*start_ends))
        logger.debug(f'NEs collected. Total {len(all_forms)} tokens processed.')
        # define linker and do linking -> get URIs
        if config['entity_linking']['linker'] == 'wikidata':
            linker = WikidataLinker(
                language=language,
                el_url=config['entity_linking']['el_url']
            )
        elif config['entity_linking']['linker'] == 'dummy':
            linker = DummyLinker()
        else:
            raise ValueError(f"Linker {config['entity_linking']['linker']} is not implemented.")
        uris = []
        for ne, si, ei, cxt in zip(nes, start_inds, end_inds, contexts):
            uri = linker.link_within_context(surface_form=ne,
                                             start_offset=si,
                                             end_offset=ei,
                                             context=cxt)
            uris.append(uri)
        # save results
        ner_cxt_lines = ['\t'.join(['NE::tag', 'Start offset', 'End offset', 'Context', 'URI'])]
        ner_cxt_lines += ['\t'.join([f'{ne}::{tag}', str(si), str(ei), cxt, uri])
                          for ne, tag, cxt, si, ei, uri in zip(nes, tags, contexts, start_inds, end_inds, uris)]
        with open(config['wikiner']['ner_contexts_output'], 'w') as f:
            f.write('\n'.join(ner_cxt_lines))
    # aggregate NEs based on URIs
    ne_aggregate = defaultdict(list)
    for i, uri in enumerate(uris):
        ne_aggregate[f'{uri}'].append(i)
    unique_ne_phrases = set(nes)
    unique_ne_tag_pairs = set(zip(nes, tags))
    logger.info(f'Total {len(nes)} NE occurrences, {len(ne_aggregate)} unique URIs, '
                f'{len(unique_ne_phrases)} unique NE surface forms, '
                f'{len(unique_ne_tag_pairs)} unique NE::tag pairs.')
    # TODO: loop over m values in $ms
    #

    induction_lang = 'eng' if language == 'en' else 'deu' if language == 'de' else None
    senses_iter = iter_senses(ne_aggregate, contexts, list(zip(start_inds, end_inds)),
                              lang=induction_lang, cxts_limit=50, n_pred=2*m,
                              n_sense_descriptors=k,
                              target_pos='N',
                              th_att_len=4,
                              logger=logger)
    ners_dict = dict()
    for j, (ner_form, ner_senses) in enumerate(senses_iter):
        if len(ner_senses) > 1:
            for i, sense in enumerate(ner_senses):
                ners_dict[f'{ner_form}##{i}'] = list(sense)
        else:
            ners_dict[f'{ner_form}'] = list(ner_senses[0])
        if j == 10:
            intermediate_file_path = ners_predictions_output_stem.parent / f'{ners_predictions_output_stem}{j}.json'
            with open(intermediate_file_path, 'w') as f:
                json.dump(ners_dict, f, indent=2)
    with open(str(ners_predictions_output_stem) + f'_m{m}_k{max_k}.json', 'w') as f:
        json.dump(ners_dict, f, indent=2)
    assert False

    # for ner_form, ner_senses in senses_iter:
    #     yield ner_form, ner_senses
    #


    # ners_dict = get_ners_dict(dataset=data,
    #                           fields=('form', '1', '2', 'tag'),
    #                           t_limit=t_limit,
    #                           ners_pred_target_folder=config['ners_predictions_output'],
    #                           lang='eng')
    # with (ners_pred_target_folder / f'preds237.json').open() as f:
    #     ners_dict = json.load(f)
    # cxt = get_cxt(ners_dict)
    # print(f'Binary matrix prepared. NEs: {len(cxt.objects)}, '
    #       f'descriptors: {len(cxt.attributes)}')
    # cxt = clean_cxt(cxt, min_att_extent=3, min_att_len=4)
    # print(f'Binary matrix cleaned. All descriptors with less than 3 corresponding entities removed for speedup. '
    #       f'descriptors: {len(cxt.attributes)}, '
    #       f'NEs: {len(cxt.objects)}, ')
    # for i, factor_dict in enumerate(iter_factors(cxt, min_atts_and_objs=3)):
    #     with (ners_pred_target_folder / 'new_classes' / f'class{i}.json').open('w') as f:
    #         json.dump(factor_dict, f)
    #
    # th_ent = 20
    # classes_path = ners_pred_target_folder / 'new_classes'
    # all_classes, filenames = read_json_classes(folder=classes_path)

    # chosen_factors, chosen_fns = zip(*[
    #     (cls, fn) for cls, fn in zip(all_classes, filenames)
    #     if len(cls[1]) > th_ent
    # ])
    # print(f'Total {len(chosen_factors)} new classes with more than {th_ent} '
    #       f'entities are taken for further analysis. Their filenames: {", ".join(filenames)}.')
    # print(f'Now we retag all the entities to distirbute them into those {len(chosen_factors)} classes.')
    # aggressive(cxt, [x[0] for x in chosen_factors], outpath=ners_pred_target_folder / 'aggressive.txt')
    # cautious(cxt, [x[0] for x in chosen_factors], outpath=ners_pred_target_folder / 'cautious.txt')

