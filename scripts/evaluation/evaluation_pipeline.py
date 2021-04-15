import json
import os.path as op
from os import scandir

from matplotlib import pylab as P

from linking.wikidata_linker import WikidataLinker
from linking.dummy_linker import DummyLinker
from linking.NE_linker import NELinker
from utils import config, this_dir, collect_entities, logger, get_params_from_dirname, config_paths
from broader_analysis import link_and_find_broaders, link_and_find_all_broaders, best_broaders
from utils import create_random_candidates
from utils import oddsratios_probs_vs_random, oddsratios_from_mean_of_random
from utils import load_candidates
from plotting import plot_against_randomized
import copy

num_random = 100
linkers = [
    # DummyLinker(),
    NELinker(config_path=config_paths[0]),
    WikidataLinker()
]
dev_thrs = 2
do_plots = True

margin = 12

induction_result_directory = config['experiment']['new_types_output_folder']
evaluation_output_directory = config['evaluation']['outputfolder']

eval_results = []

m_dirs = [f.path for f in scandir(induction_result_directory) if f.is_dir()]
cols_for_output = ["k", "m", "th",
                   "linker",
                   "oddsratios_probs_vs_random_LIB", "above2std_of_oddsrations_vs_random_HIB"]

with open(op.join(evaluation_output_directory, "results.csv"), "w") as fout:
    fout.write("\t".join(cols_for_output) + "\n")
    for mdir in m_dirs:
        experiment_dirs = [f.path for f in scandir(mdir) if f.is_dir()]
        for exp_dir in experiment_dirs:
            param_dict = get_params_from_dirname(exp_dir)
            pattern = "k" + str(param_dict["k"]) + "_" + "th" + str(param_dict["th"]) + "_type%d.json"
            induced_candidates_ = load_candidates(exp_dir, pattern=pattern)
            if len(induced_candidates_) == 0:
                logger.error(str(param_dict) + " has an empyt directory: " + str(exp_dir))
                break
            all_entities = collect_entities(induced_candidates_)
            # Generating randoms
            randomized_candidates_ = create_random_candidates(induced_candidates_,
                                                              num_random + margin)
            for linker in linkers:
                # Linking results
                induced_candidates = copy.deepcopy(induced_candidates_)
                randomized_candidates = copy.deepcopy(randomized_candidates_)
                broaders_for_all_entities = link_and_find_all_broaders(all_entities, linker)
                per_candidate_broader_counts = link_and_find_broaders(induced_candidates, linker)

                log_odds_of_induced = best_broaders(broaders_for_all_entities,
                                                    per_candidate_broader_counts)
                # print(log_odds_of_induced,"\n--")

                # Generate random equivalents and find their broaders ------

                randomized_logodds = []
                for rcani, rcan in enumerate(randomized_candidates):
                    rcan_links_and_supers = link_and_find_broaders(rcan, linker)
                    if any([len(rcan['broader_counts']) < 1 for rcan in rcan_links_and_supers]):
                        continue
                    rcan_log_odds = best_broaders(broaders_for_all_entities,
                                                  rcan_links_and_supers)

                    randomized_logodds.append(rcan_log_odds)
                if len(randomized_logodds) > num_random:
                    randomized_logodds = randomized_logodds[:num_random]


                # Quantitative evaluations ---------------------------------

                quant1 = oddsratios_probs_vs_random(log_odds_of_induced,
                                                    randomized_logodds)

                quant2 = oddsratios_from_mean_of_random(log_odds_of_induced,
                                                        randomized_logodds,
                                                        dev_thrs=dev_thrs)

                if do_plots:
                    paramkeys = list(param_dict.keys())
                    paramkeys.sort()
                    paramstr = "_".join([k+str(param_dict[k]) for k in paramkeys])
                    figsuf = "_" + str(paramstr) + "_" + str(linker.__class__.__name__)
                    figfilename = op.join(evaluation_output_directory, "LOGODDS_" + figsuf + ".png")
                    fig_extratext="pval: "+("%.4f" % quant1)+"  outl: "+str(quant2)
                    plot_against_randomized(log_odds_of_induced,
                                            randomized_logodds,
                                            figsufix=figsuf, extratext=fig_extratext)
                    P.savefig(figfilename)

                this_res = dict()
                this_res.update(param_dict)
                this_res["linker"] = linker.__class__.__name__
                this_res["oddsratios_probs_vs_random_LIB"] = quant1
                this_res["std_of_oddsrations_vs_random_HIB"] = quant2
                this_res["number_of_candidates"] = len(induced_candidates)

                eval_results.append(this_res)
                print(json.dumps(this_res, indent=2))
                fout.write("\t".join([str(this_res[x])
                                      for x in cols_for_output])+"\n")
        fout.flush()

        try:
            linker._write_cache()
        except:
            pass