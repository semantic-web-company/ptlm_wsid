import json
import logging, logging.config
from pathlib import Path

import configparser
from tqdm import tqdm

import fca
from ptlm_wsid.utils import get_cxt, clean_cxt


if __name__ == '__main__':
    def iter_factors(cxt, min_descriptors, fidelity=0.5):
        """
        Fidelity is the fraction of crosses in the binary matrix `cxt` that have to be covered by the output factors.
        The higher this value is, the more factors will be produced. Large factors are found first, the subsequent factors
        decrease in size, therefore larger values of fidelity will yield larger number of insignificant factors in the tail
        of the output. Fidelity does not it influence the order of discovery.
        """
        factors_iter = fca.algorithms.factors.algorithm2_w_condition(
            cxt,
            fidelity=fidelity,
            allow_repeatitions=False,
            min_atts_and_objs=min_descriptors,
            objs_ge_atts=True)
        ###
        pbar = tqdm(enumerate(factors_iter), total=fidelity)
        pbar.set_description(desc=f'Inducing new types with k={k} and at least {min_descriptors} descriptors ')
        prev_agg_score = 0
        for i, (factor, cls_score, agg_score) in enumerate(factors_iter):
            out = f'\nType suggestion {i}.\n'
            out += f'{", ".join(factor.intent)}.\n'
            out += f'Type score: {cls_score:0.4f}, aggregated score: {agg_score:0.4f}\n'
            out += f'Total senses: {len(factor.extent)}, total descriptors: {len(factor.intent)}'
            logger.debug(out)
            out_json = {
                'score': cls_score,
                'descriptors': list(factor.intent),
                'entities': list(factor.extent)
            }
            pbar.update(agg_score - prev_agg_score)
            prev_agg_score = agg_score
            yield out_json

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
    ners_predictions_path = Path(config['wikiner']['ners_predictions_path'])
    language = config['wikiner']['language']
    new_types_output_folder = Path(config['wikiner']['new_types_output_folder'])
    ks = [int(k.strip()) for k in config['wikiner']['ks'].split(',')]
    ths_n_descriptors = [int(x.strip()) for x in config['wikiner']['ths_type_descriptors'].split(',')]
    # Read data
    with open(ners_predictions_path) as f:
        ners_dict = json.load(f)

    for k in ks:
        k_ners_dict = {ent: sense_names[:k] for ent, sense_names in ners_dict.items()}
        cxt = get_cxt(k_ners_dict)
        logger.info(f'Binary matrix prepared. k: {k}, NEs: {len(cxt.objects)}, descriptors: {len(cxt.attributes)}')
        min_extent = 3
        cxt = clean_cxt(cxt, min_att_extent=min_extent,
                        # min_att_len=4
                        )
        logger.info(f'Binary matrix cleaned. All descriptors with less than {min_extent} corresponding entities '
                    f'removed for speedup. descriptors: {len(cxt.attributes)}, NEs: {len(cxt.objects)}, ')
        for th_descrs in ths_n_descriptors:
            parametrized_folder = new_types_output_folder / f'k{k}_th{th_descrs}'
            parametrized_folder.mkdir(exist_ok=True)
            for i, factor_dict in enumerate(iter_factors(cxt, min_descriptors=th_descrs)):
                with (parametrized_folder / f'k{k}_th{th_descrs}_type{i}.json').open('w') as f:
                    json.dump(factor_dict, f, indent=2)
