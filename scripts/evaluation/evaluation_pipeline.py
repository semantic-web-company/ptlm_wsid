import json
import os.path as op
from os import scandir
import copy

from matplotlib import pylab as P
import numpy as np

from linking.wikidata_linker import WikidataLinker
from linking.NE_linker import NELinker
from utils import config, this_dir, collect_entities, logger, get_params_from_dirname, config_paths
from broader_analysis import link_and_find_broaders, link_and_find_all_broaders, best_broaders, add_labels_to_supers
from utils import create_random_candidates
from utils import oddsratios_probs_vs_random, oddsratios_from_mean_of_random, kl_vs_random
from utils import load_candidates
from utils import build_uri2surfaceform
from plotting import plot_against_randomized

num_random = 100  # Number of random candidates generated for eval purposes.
dev_thrs = 2  # Candidates more than this number of std from the random-mean are counted as good.
do_plots = True
min_sizes = {"en": 10, "de": 20}

induction_result_directory = config['experiment']['new_types_output_folder']
evaluation_output_directory = config['evaluation']['outputfolder']
language = config['wikiner']['language']
linkedtsvfile = config['wikiner']['ner_contexts_output']

linkers = [
    # DummyLinker(),
    NELinker(config_path=config_paths[0]),
    WikidataLinker(language=language)
]
margin = 12

eval_results = []
uri2surfaceform = build_uri2surfaceform(linkedtsvfile)

m_dirs = [f.path for f in scandir(induction_result_directory) if f.is_dir()]
cols_for_output = ["k", "m", "th",
                   "linker",
                   "oddsratios_probs_vs_random_LIB",
                   "above2std_of_oddsrations_vs_random_HIB",
                   "KL_vs_random_HIB",
                   "number_of_candidates",
                   "average_candidate_size",
                   "maximum_candidate_size",
                   "minumum_candidate_size"]

outfn = language + "_" + str(min_sizes[language]) + "_results.csv"
with open(op.join(evaluation_output_directory, outfn), "w") as fout:
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
            induced_candidates_ = [can for can in induced_candidates_
                                   if (len(can["entities"])) >= min_sizes[language]]
            if len(induced_candidates_) < 2:
                logger.error("\n\nToo few candidates left for  " + str(param_dict))
                continue
            all_entities = collect_entities(induced_candidates_)
            # Generating randoms
            randomized_candidates_ = create_random_candidates(induced_candidates_,
                                                              num_random + margin)
            for can in induced_candidates_:
                can["entity_labels"] = [uri2surfaceform.get(x, [x])[0] for x in can["entities"]]
            for linker in linkers:
                # Linking results
                induced_candidates = copy.deepcopy(induced_candidates_)
                randomized_candidates = copy.deepcopy(randomized_candidates_)
                broaders_for_all_entities = link_and_find_all_broaders(all_entities, linker)
                per_candidate_broader_counts = link_and_find_broaders(induced_candidates, linker)

                log_odds_of_induced = best_broaders(broaders_for_all_entities,
                                                    per_candidate_broader_counts)
                best_labels = add_labels_to_supers(per_candidate_broader_counts,
                                                   linker)
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
                quant3 = kl_vs_random(log_odds_of_induced, randomized_logodds)

                if do_plots:
                    paramkeys = list(param_dict.keys())
                    paramkeys.sort()
                    paramstr = "_".join([k + str(param_dict[k]) for k in paramkeys])
                    figsuf = "_" + str(paramstr) + "_" + str(linker.__class__.__name__)
                    figfilename = op.join(evaluation_output_directory, "LOGODDS_" + figsuf + ".png")
                    fig_extratext = "pval: " + ("%.4f" % quant1) + "  outl: " + str(quant2) + " kl:" + ("%.4f" % quant3)
                    plot_against_randomized(log_odds_of_induced,
                                            randomized_logodds,
                                            figsufix=figsuf, extratext=fig_extratext)
                    P.savefig(figfilename)
                    P.close()

                can_sizes = [len(x['entities']) for x in induced_candidates_]
                this_res = dict()
                this_res.update(param_dict)
                this_res["linker"] = linker.__class__.__name__
                this_res["oddsratios_probs_vs_random_LIB"] = quant1
                this_res["above2std_of_oddsrations_vs_random_HIB"] = quant2
                this_res["KL_vs_random_HIB"] = quant3
                this_res["number_of_candidates"] = len(induced_candidates_)
                this_res["average_candidate_size"] = np.mean(can_sizes)
                this_res["maximum_candidate_size"] = max(can_sizes)
                this_res["minumum_candidate_size"] = min(can_sizes)

                eval_results.append(this_res)
                # print(json.dumps(this_res, indent=2))
                fout.write("\t".join([str(this_res[x])
                                      for x in cols_for_output]) + "\n")
            print("Done: ", str(param_dict))
        fout.flush()

        try:
            linker._write_cache()
        except:
            pass
