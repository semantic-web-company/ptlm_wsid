import json
import logging, logging.config
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import configparser
import conllu

from ptlm_wsid.chi import collect_ners, iter_senses
from linking.dummy_linker import DummyLinker
from linking.wikidata_linker import WikidataLinker


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
    nes_senses_output_stem = Path(config['wikiner']['nes_senses_output_stem'])
    nes_in_cxts_path = Path(config['wikiner']['ner_contexts_output'])
    language = config['wikiner']['language']
    t_limit = int(config['wikiner']['tokens_to_be_processed'])
    conll_data_path = config['wikiner']['conll_data_path']
    ms = [int(m) for m in config['wikiner']['ms'].split(',')]
    max_k = int(config['wikiner']['max_k'])
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
    #
    for m in ms:
        induction_lang = 'eng' if language == 'en' else 'deu' if language == 'de' else None
        logger.info(f'Starting to identify senses for m={m}.')
        senses_iter = iter_senses(ne_aggregate, contexts, list(zip(start_inds, end_inds)),
                                  lang=induction_lang, cxts_limit=50, n_pred=2*m,
                                  n_sense_descriptors=max_k,
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
                intermediate_file_path = nes_senses_output_stem.parent / f'{nes_senses_output_stem}_m{m}_k{max_k}_{j}.json'
                with open(intermediate_file_path, 'w') as f:
                    json.dump(ners_dict, f, indent=2)
        with open(str(nes_senses_output_stem) + f'_m{m}_k{max_k}.json', 'w') as f:
            json.dump(ners_dict, f, indent=2)
