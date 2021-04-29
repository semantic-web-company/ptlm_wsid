import json
import os.path as op
from os import scandir
import copy

from matplotlib import pylab as P
import numpy as np

from linking.wikidata_linker import WikidataLinker
from linking.NE_linker import NELinker
from utils import config, this_dir, collect_entities, logger, get_params_from_dirname, config_paths
from broader_analysis import link_and_find_broaders, link_and_find_all_broaders, add_labels_to_supers
from broader_analysis import best_broaders_enrichment_suprise as best_broaders
from broader_analysis import binocache
from utils import create_random_candidates
from utils import oddsratios_probs_vs_random, oddsratios_from_mean_of_random, kl_vs_random
from utils import load_candidates
from utils import build_uri2surfaceform
from plotting import plot_against_randomized

num_random = 0  # Number of random candidates generated for eval purposes.
dev_thrs = 2  # Candidates more than this number of std from the random-mean are counted as good.
do_plots = True
min_sizes = {"en": 0, "de": 0}
pvalthrs = 0.05

induction_result_directory = config['experiment']['new_types_output_folder']
evaluation_output_directory = config['evaluation']['outputfolder']
language = config['wikiner']['language']
linkedtsvfile = config['wikiner']['ner_contexts_output']

linkers = [
    # DummyLinker(),
    NELinker(config_path=config_paths[0]),
    WikidataLinker(language=language)
]
margin = 1

eval_results = []
uri2surfaceform = build_uri2surfaceform(linkedtsvfile)

m_dirs = [f.path for f in scandir(induction_result_directory) if f.is_dir()]
m_dirs.sort(key=lambda x: int(x.split("/")[-1][1:]))
# m_dirs.reverse()
cols_for_output = ["k", "m", "th",
                   "linker",
                   "oddsratios_probs_vs_random_LIB",
                   "above2std_of_oddsrations_vs_random_HIB",
                   "KL_vs_random_HIB",
                   "number_of_candidates",
                   "average_candidate_size",
                   "maximum_candidate_size",
                   "minumum_candidate_size",
                   "proportion_significantly_enriched_HIB",
                   "proportion_matching_smaller_than_average_broaders_HIB",
                   "max_pvalue"
                   ]

bcach = binocache()
wholecsvfilename = "WHOLE"+language + "_" + str(min_sizes[language]) + "_results.csv"
outfn = language + "_" + str(min_sizes[language]) + "_results.csv"
#wholeout = open(wholecsvfilename, "w")
with open(op.join(evaluation_output_directory, outfn), "w") as fout:
    fout.write("\t".join(cols_for_output) + "\n")
    for mdir in m_dirs:
        experiment_dirs = [f.path for f in scandir(mdir) if f.is_dir()]
        # experiment_dirs.sort(key=lambda x:x.split("_")[-1])
        for exp_dir in experiment_dirs:
            param_dict = get_params_from_dirname(exp_dir)
            pattern = "k" + str(param_dict["k"]) + "_" + "th" + str(param_dict["th"]) + "_type%d.json"
            induced_candidates_ = load_candidates(exp_dir, pattern=pattern)
            if len(induced_candidates_) == 0:
                logger.error(str(param_dict) + " has an empyt directory: " + str(exp_dir))
                break
            
            induced_candidates_ = [can for can in induced_candidates_
                                   if len(can["entities"])  >= min_sizes[language]]
            if len(induced_candidates_) < 2:
                logger.error("\n\nToo few candidates left for  " + str(param_dict))
                continue
            all_entities = collect_entities(induced_candidates_)
            # Generating randoms
            randomized_candidates_ = []
            if num_random > 0:
                randomized_candidates_ = create_random_candidates(induced_candidates_,
                                                                  num_random + margin)
            for can in induced_candidates_:
                can["entity_labels"] = [uri2surfaceform.get(x.split("##")[0], [x])[0] for x in can["entities"]]
            for linker in linkers:
                # Linking results
                induced_candidates = copy.deepcopy(induced_candidates_)
                randomized_candidates = copy.deepcopy(randomized_candidates_)
                broaders_for_all_entities = link_and_find_all_broaders(all_entities, linker)
                per_candidate_broader_counts = link_and_find_broaders(induced_candidates, linker)

                pavals_for_induced = best_broaders(broaders_for_all_entities,
                                                   per_candidate_broader_counts,
                                                   pvalthrs=pvalthrs,
                                                   bcach=bcach)
                print(linker.__class__.__name__,
                      "_".join(exp_dir.split("/")[-2:]),
                      ", ".join([str(len(can["log_odds"])) for can in per_candidate_broader_counts]))

                best_labels = add_labels_to_supers(per_candidate_broader_counts,
                                                   linker)
                # print(log_odds_of_induced,"\n--")

                # Generate random equivalents and find their broaders ------

                randomized_p_vals = []
                for rcani, rcan in enumerate(randomized_candidates):
                    rcan_links_and_supers = link_and_find_broaders(rcan, linker)
                    if any([len(rcan['broader_counts']) < 1 for rcan in rcan_links_and_supers]):
                        continue
                    rcan_log_odds = best_broaders(broaders_for_all_entities,
                                                  rcan_links_and_supers,
                                                  pvalthrs=pvalthrs,
                                                  bcach=bcach)

                    randomized_p_vals.append(rcan_log_odds)
                if len(randomized_p_vals) > num_random:
                    randomized_p_vals = randomized_p_vals[:num_random]

                # Quantitative evaluations ---------------------------------
                quant1 = quant2 = quant3 = 0
                if len(randomized_p_vals) > 0:
                    quant1 = oddsratios_probs_vs_random(pavals_for_induced,
                                                        randomized_p_vals)

                    quant2 = oddsratios_from_mean_of_random(pavals_for_induced,
                                                            randomized_p_vals,
                                                            dev_thrs=dev_thrs)
                    quant3 = kl_vs_random(pavals_for_induced, randomized_p_vals)

                numcans = float(len(pavals_for_induced))
                # Number of candidates which are significantly over represented in a candidate broader
                quant4 = len([x for i, x in enumerate(pavals_for_induced)
                              if x is not None
                              and x < (pvalthrs / (  len(per_candidate_broader_counts[i]["broader_counts"]) )
                                       )]) / numcans

                # Proportion of candidates that best-match categories which are less than average size
                quant5 = len([p for p in per_candidate_broader_counts
                              if len(p["log_odds"]) > 0
                              and p["log_odds"][0]["loggods"] < (pvalthrs / len(p["broader_counts"]))
                              and p["log_odds"][0]["suprise"] < 1.0]) / numcans

                # Maximum p-value
                quant6 = np.max([x if x is not None else pvalthrs*2
                                 for x in pavals_for_induced])

                print("\t", quant4, quant5)

                if do_plots and len(randomized_p_vals) > 0:
                    paramkeys = list(param_dict.keys())
                    paramkeys.sort()
                    paramstr = "_".join([k + str(param_dict[k]) for k in paramkeys])
                    figsuf = "_" + str(paramstr) + "_" + str(linker.__class__.__name__)
                    figfilename = op.join(evaluation_output_directory, "LOGODDS_" + figsuf + ".png")
                    fig_extratext = "pval: " + ("%.4f" % quant1) + "  outl: " + str(quant2) + " kl:" + ("%.4f" % quant3)
                    plot_against_randomized(pavals_for_induced,
                                            randomized_p_vals,
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
                this_res["above2std_of_oddsrations_vs_random_normalized_HIB"] = quant2 / float(len(induced_candidates_))
                this_res["one_minus_avg_pval_oddsratios_HIB"] = 1 - quant1
                this_res["proportion_significantly_enriched_HIB"] = quant4
                this_res["proportion_matching_smaller_than_average_broaders_HIB"] = quant5
                this_res["max_pvalue"] = quant6

                eval_results.append(this_res)
                # print(json.dumps(this_res, indent=2))
                fout.write("\t".join([str(this_res[x])
                                      for x in cols_for_output]) + "\n")

                # Write JSONs
                for can in per_candidate_broader_counts:
                    can["size"] = len(can['entities'])
                eval_params = {"linker": linker.__class__.__name__,
                               "num_random": num_random,
                               "min_sizes": min_sizes[language],
                               "dev_thrs": dev_thrs,
                               "pvalthrs": pvalthrs}
                jsondir = op.join(evaluation_output_directory, "jsons_" + language)
                jsonname = linker.__class__.__name__ + "__" + "_".join(exp_dir.split("/")[-2:]) + ".json"
                outputjson = {"candidate_types": per_candidate_broader_counts,
                              "evaluation_results": this_res,
                              "evaluation_parameters": eval_params}
                if min_sizes[language]>0:
                    with open(op.join(jsondir, jsonname), "w") as jfout:
                        json.dump(outputjson, jfout, indent=2)
                        jfout.close()
                else:
                    pass


            # print("Done: ", str(param_dict))

        fout.flush()

        try:
            linker._write_cache()
        except:
            pass
