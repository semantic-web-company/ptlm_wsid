import json
from collections import defaultdict

from pathlib import Path

import fca
from fca.algorithms.factors import algorithm2

from ptlm_wsid.target_context import nlp_dict

work_folder = Path('./geo_all_separate_goodpos')


def normalize_cxt(cxt: fca.Context):
    nlp_de, lang = nlp_dict['de']
    norm_atts = defaultdict(set)
    for att in cxt.attributes:
        # r = nlp_de.tokenizer(att)
        r = nlp_de(att[0].upper() + att[1:])
        norm_atts[r[0].lemma_.lower()].add(att)
    for new_att, old_atts in norm_atts.items():
        if len(old_atts) > 1:
            old_extents = [cxt.get_attribute_extent(a) for a in old_atts]
            new_extent = set()
            for ext in old_extents:
                new_extent |= ext
            assert len(new_extent) >= len(old_extents[0])
            try:
                cxt.set_attribute_extent(new_extent, new_att)
            except KeyError:
                cxt.add_attribute_with_extent(new_extent, new_att)
            cxt.delete_attributes([a for a in (old_atts - {new_att})])
    print(f'Objects: {len(cxt.objects)}, attributes: {len(cxt.attributes)}, '
          f'density: {sum(sum(cxt.table)) / (len(cxt.objects) * len(cxt.attributes))}')
    return cxt


def write_cxt(cxt_target_path: Path = work_folder / 'ners_context.cxt',
              ners_json_path: Path = work_folder / 'ners.json',):
    with ners_json_path.open():
        ners_dict = json.loads(ners_json_path.read_text())
    objects = list(ners_dict.keys())
    attributes = list(set(sum(ners_dict.values(), [])))
    table = [[att in ners_dict[obj] for att in attributes] for obj in objects]
    cxt = fca.Context(objects=objects, attributes=attributes, cross_table=table)
    print(f'Objects: {len(cxt.objects)}, attributes: {len(cxt.attributes)}, '
          f'density: {sum(sum(cxt.table))/(len(cxt.objects)*len(cxt.attributes))}')
    cxt = normalize_cxt(cxt)
    fca.write_cxt(cxt, cxt_target_path)


def write_cxts(cxt_target_folder: Path = work_folder,
               ners_json_path: Path = work_folder / 'ners.json',
               ontology=None):
    with ners_json_path.open():
        ners_dict = json.loads(ners_json_path.read_text())
    if ontology is None:
        ontology = dict()

    obj2cls = dict()
    for obj in ners_dict:
        splitted = obj.split(', ')
        name = ', '.join(splitted[:-1])
        cls = splitted[-1]
        name = name.lstrip('(')
        cls = cls.rstrip(')')
        obj2cls[name] = cls
    # cxts = {cls: dict() for cls in set(ontology.values())}
    cxts = dict()
    for obj, cls in obj2cls.items():
        obj_str = ', '.join((obj, cls))
        if cls not in ontology:
            ontology[cls] = cls
            cxts[cls] = dict()
        cxts[ontology.get(cls, cls)][obj_str] = ners_dict[obj_str]
    stats = '\n'.join(f'{cls}: {len(v)}' for cls, v in cxts.items())
    print(stats)

    for cls in cxts:
        cls_dict = cxts[cls]
        objects = list(cls_dict.keys())
        attributes = list(set(sum(cls_dict.values(), [])))
        table = [[att in cls_dict[obj] for att in attributes] for obj in objects]
        cxt = fca.Context(objects=objects, attributes=attributes, cross_table=table)
        print(f'{cls}\n'
              f'Objects: {len(cxt.objects)}, attributes: {len(cxt.attributes)}, '
              f'density: {sum(sum(cxt.table))/(len(cxt.objects)*len(cxt.attributes))}')
        cxt = normalize_cxt(cxt)
        fca.write_cxt(cxt, cxt_target_folder / (cls + '.cxt'))
    return ontology


def get_factors(cxt_path: Path = work_folder / 'ners_context.cxt',
                factors_target_path: Path = work_folder / 'factors.json',
                fidelity=0.5):
    cxt = fca.read_cxt(cxt_path)
    factor_iter = algorithm2(cxt, fidelity)
    factors_out = []
    acc_score = 0
    for i, (factor, score) in enumerate(factor_iter):
        if len(factor.extent) > len(factor.intent):
            factors_out.append( (list(factor.intent), list(factor.extent)) )
            acc_score += score
            print(f'#{i} {acc_score}\n'
                  f'Score: {score}\n'
                  f'Intent: {len(factor.intent)}\n'
                  f'Extent: {len(factor.extent)}\n')
        else:
            print(f'#{i} len(extent) = {len(factor.extent)}, len(intent) = {len(factor.intent)}')
    with factors_target_path.open('w'):
        out = json.dumps(factors_out, indent=2)
        factors_target_path.write_text(out)


def get_higher_level_factor(factors_path: Path = work_folder / 'factors.json',
                            refined_factors_target_path: Path = work_folder / 'factors_refined.json'):
    with factors_path.open():
        factors = json.loads(factors_path.read_text())
    intents = [x[0] for x in factors]
    hierarchy = defaultdict(list)
    for intent in intents:
        subclss = [tuple(other_intent)
                   for other_intent in intents
                   if set(other_intent) > set(intent)]
        if subclss:
            for cls_ in subclss:
                hierarchy[cls_].append(intent)
    higher_level_clss = [factor for factor in factors if tuple(factor[0]) not in hierarchy]
    with refined_factors_target_path.open('w'):
        out = json.dumps(higher_level_clss, indent=2)
        refined_factors_target_path.write_text(out)


ontology = dict(
    RR='PER',
    AN='PER',
    PER='PER',
    #
    LD='LOC',
    LDS='LOC',
    ST='LOC',
    STR='LOC',
    # LOC='LOC',
    #
    ORG='ORG',
    UN='ORG',
    GRT='ORG',
    INN='ORG',
    MRK='ORG',
    #
    GS='DOC',
    VO='DOC',
    EUN='DOC',
    NRM='DOC',
    VS='DOC',
    VT='DOC',
    REG='DOC',
    RS='DOC',
    LIT='DOC',
)


if __name__ == '__main__':
    def do_many_cxts():
        ontology = write_cxts()
        for cls in set(ontology.values()):
            print(f'Class: {cls}')
            cxt_path = work_folder / (cls + '.cxt')
            factors_path = work_folder / (cls + '_factors.json')
            refined_factors_path = work_folder / (cls + '_factors_refined.json')
            get_factors(cxt_path=cxt_path, factors_target_path=factors_path)
            get_higher_level_factor(factors_path=factors_path,
                                    refined_factors_target_path=refined_factors_path)
            output_path = work_folder / (cls + '_compound_factors.json')
            get_compound_factors(factors_path, output_path)

    def do_one_cxt():
        write_cxt()
        get_factors()
        get_higher_level_factor()

    def get_compound_factors(factors_path: Path = work_folder / 'factors.json',
                             output_path: Path = work_folder / 'compound_factors.json'):
        with factors_path.open():
            factors = json.loads(factors_path.read_text())
        out = []
        for factor in factors:
            if len(factor[0]) > 1:
                out.append(factor)

        with output_path.open('w'):
            out = json.dumps(out, indent=2)
            output_path.write_text(out)

    do_many_cxts()
