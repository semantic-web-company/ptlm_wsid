from typing import List, Dict, Set
import random as ran
import logging


def create_random_candidates(induced_candidates: List[Dict],
                             num_random: int):
    """

    @param induced_candidates: a list of dictionaries respresenting candidates. Each must have "entities" and "descriptors"
    @param num_random: number of random replicates
    @return: a list of num_random randomizations of the candidate classes.
             Each randomization is a list of candidates equal in dimensions to the induced_candidates
             but with entities asigned at random.
             Notice that if an entity appears in two different candidates in induced_candidates
             it will also appear in two different candidates in each randomization
    """
    candidate_sizes = [len(ic["entities"]) for ic in induced_candidates]
    result = []
    all_entities_constant = []
    for ic in induced_candidates:
        all_entities_constant += ic["entities"]

    for nr in range(num_random):
        all_entities = all_entities_constant.copy()
        this_replicate = []
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
                    logging.error("coundn't find a replicate after "+str(numattemps))
            all_entities = all_entities[siz:]
            di = {"entities": ents,
                  "descriptors": ["replicate_" + str(nr),
                                  "candidate_" + str(sn)]}
            this_replicate.append(di)
        result.append(this_replicate)
    return result
