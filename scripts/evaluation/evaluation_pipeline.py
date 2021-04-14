import json
import os.path as op
from os import scandir

from matplotlib import pylab as P

from linking.wikidata_linker import WikidataLinker
from linking.dummy_linker import DummyLinker
from utils import config, this_dir, collect_entities, logger, get_params_from_dirname
from broader_analysis import link_and_find_broaders, link_and_find_all_broaders, best_broaders
from utils import create_random_candidates
from utils import oddsratios_probs_vs_random
from utils import  load_candidates
from plotting import plot_against_randomized

num_random = 100

induction_result_directory = config['wikiner']['new_types_output_folder']
eval_results = []


m_dirs = [f.path for f in scandir(induction_result_directory) if f.is_dir()]
for mdir in m_dirs:
    experiment_dirs = [f.path for f in scandir(mdir) if f.is_dir()]
    for exp_dir in experiment_dirs:
        param_dict = get_params_from_dirname(exp_dir)
        pattern = "k"+str(param_dict["k"])+"_"+"th"+str(param_dict["th"])+"_type%d.json"
        induced_candidates = load_candidates(exp_dir, pattern=pattern)
        if len(induced_candidates)==0:
            logger.error(str(param_dict)+" has an empyt directory: "+str(exp_dir))
            break
        all_entities = collect_entities(induced_candidates)

        for linker in [DummyLinker()]:# , WikidataLinker()]:
            # Linking results
            broaders_for_all_entities = link_and_find_all_broaders(all_entities, linker)
            per_candidate_broader_counts = link_and_find_broaders(induced_candidates, linker)

            log_odds_of_induced = best_broaders(broaders_for_all_entities,
                                                per_candidate_broader_counts)

            # Generating randoms
            randomized_candidates = create_random_candidates(induced_candidates,
                                                             num_random)



            randomized_logodds = []
            for rcan in randomized_candidates:
                rcan_links_and_supers = link_and_find_broaders(rcan, linker)
                rcan_log_odds = best_broaders(broaders_for_all_entities,
                                              rcan_links_and_supers)
                randomized_logodds.append(rcan_log_odds)

            #plot_against_randomized(log_odds_of_induced,
            #                        randomized_logodds)

            quant1 = oddsratios_probs_vs_random(log_odds_of_induced,
                                               randomized_logodds)

            this_res = dict()
            this_res.update(param_dict)
            this_res["linker"] = linker.__class__.__name__
            this_res["oddsratios_probs_vs_random_LIB"] = quant1

            eval_results.append(this_res)
            print(json.dumps(this_res, indent=2))