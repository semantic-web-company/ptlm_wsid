import os.path as op

from linking.wikidata_linker import WikidataLinker
from utils import config, this_dir, collect_entities
from broader_analysis import link_and_find_broaders, link_and_find_all_broaders, best_broaders
from randomization import create_random_candidates
from readers import  load_candidates

induction_result_directory = config['wikiner']['nes_senses_output_stem']

# Defaults for quick experiments
if not op.isdir(induction_result_directory):
    induction_result_directory = op.join(this_dir,"data/candidates/k10_th3/")


num_random = 100
induced_candidates = load_candidates(induction_result_directory)
all_entities = collect_entities(induced_candidates)

for linker in [WikidataLinker()]:
    # Linking results
    all_super_counts = link_and_find_all_broaders(all_entities, linker)
    per_candidate_super_counts = link_and_find_broaders(induced_candidates, linker)

    log_odds_of_induced = best_broaders(all_super_counts,
                                        per_candidate_super_counts)

    # Generating randoms
    randomized_candidates = create_random_candidates(induced_candidates,
                                                     num_random)



    randomized_logodds = []
    for rcan in randomized_candidates:
        rcan_links_and_supers = link_and_find_broaders(rcan, linker)
        rcan_log_odds = best_broaders(all_super_counts,
                                      rcan_links_and_supers)
        randomized_logodds.append(rcan_log_odds)

    #plot_against_randomized(log_odds_of_induced,
    #                        randomized_logodds)


