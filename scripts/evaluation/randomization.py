from typing import List, Dict, Set
import random as ran
import logging


def create_random_candidates(induced_candidates: List[Dict],
                             num_random: int):
    """

    @param induced_candidates: a list of dictionaries respresenting new_types. Each must have "entities" and "descriptors"
    @param num_random: number of random replicates
    @return: a list of num_random randomizations of the candidate classes.
             Each randomization is a list of new_types equal in dimensions to the induced_candidates
             but with entities asigned at random.
             Notice that if an entity appears in two different new_types in induced_candidates
             it will also appear in two different new_types in each randomization
    """
    candidate_sizes = [len(ic["entities"]) for ic in induced_candidates]
    result = []
    all_entities_constant = []
    for ic in induced_candidates:
        all_entities_constant += ic["entities"]

    for nr in range(num_random):
        finished = False
        while not finished:
            all_entities = all_entities_constant.copy()
            this_replicate = []
            finished = True
            ran.shuffle(candidate_sizes)
            ran.shuffle(all_entities)
            for sn, siz in enumerate(candidate_sizes):
                ents = list()
                numattemps = 0
                while len(set(ents)) != siz:
                    ran.shuffle(all_entities)
                    ents = all_entities[:siz]
                    numattemps += 1
                    if numattemps > 10:
                        finished = False
                        break
                all_entities = all_entities[siz:]
                di = {"entities": ents,
                      "descriptors": ["replicate_" + str(nr),
                                      "candidate_" + str(sn)]}
                this_replicate.append(di)
            result.append(this_replicate)
    return result
