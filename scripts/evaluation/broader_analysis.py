import math
from typing import List, Dict, Set
from linking.entity_linker import EntityLinker
from linking.utils import is_uri
from linking.dummy_linker import DummyLinker as DuLi
from scipy.stats import binom
import numpy as np


def link_and_find_broaders(candidates: List[Dict],
                           linker: EntityLinker = None):
    """

    @param candidates:
    @param linker:
    @return: a list of new_types, with an extra key called "broader_counts" which is a dictionary
    """
    result = []
    if linker is None:
        linker = DuLi()
    for can in candidates:
        broader_counts = dict()
        for ent in can["entities"]:
            broaders = linker.find_broaders(ent)
            for br in broaders:
                broader_counts[br] = broader_counts.get(br, 0) + 1
        can["broader_counts"] = broader_counts

    return candidates


def link_and_find_all_broaders(entities: Set[str],
                               linker: EntityLinker = None) -> Dict[str, List[str]]:
    """
    For all entities, returns a list of broader categories to each.
    If an entity is a URI, the linker is used only to query for broaders
    If an entity is a string, the linker is used to link, and the best
    matching entity is used for broaders
    
    @param entities: set of entities, each being a string, optionally a URI
    @return: dictionary that defines, for each of the original entities, a list of uris of broader types
    """
    if linker is None:
        linker = DuLi()
    result = dict()
    for ent in entities:
        broaders = linker.find_broaders(uri=ent)
        result[ent] = broaders

    return result


def add_labels_to_supers(per_candidate_links_and_supersList: List[Dict],
                         linker: EntityLinker):
    all_labels = []
    for can in per_candidate_links_and_supersList:
        bestsuper = can["best_match_broader"]
        bestsuper_label = ""
        if bestsuper is not None:
            bestsuper_label = linker.find_label(bestsuper)
        can["best_match_broader_label"] = bestsuper_label
        all_labels.append(bestsuper_label)
        for cbroader in can["log_odds"]:
            cbroader["candidatebroaderlabel"] = linker.find_label(cbroader["candidatesbroader"])
    return all_labels


def pval(can_size, prop_broad_in_corpus, common_in_can_and_prop, cache=None):
    """
    The p-value of a broader being over/under represented in a candidate.
    The computation is based on a binomial null-model as specified in:
      Mi, H., Muruganujan, A., Casagrande, J. T., & Thomas, P. D. (2013).
      Large-scale gene function analysis with the PANTHER classification system.
      Nature protocols, 8(8), 1551-1566.
      doi: 10.1038/nprot.2013.092

    @param can_size:  size of the candidate
    @param prop_broad_in_corpus:   proportion (between 0 and 1) of all the entitites that belong to this broader
    @param common_in_can_and_prop:   number of common entitites between the broader and the candidate
    @return:  positive if over-represented, negative if under-represented
    """
    Ti = can_size
    Pcj = float(prop_broad_in_corpus)
    Nij = common_in_can_and_prop
    if (Ti,Pcj,Nij) in cache.cach.keys():
        return cache[(Ti,Pcj,Nij)]

    if (Ti,Pcj) in cache.cach.keys():
        b = cache[(Ti,Pcj)]
    else:
        b=binom(Ti, Pcj)



    pval = 0
    if Nij / float(Ti) > Pcj:
        for k in range(common_in_can_and_prop, Ti + 1):
            pval += b.pmf(k)
    else:
        for k in range(common_in_can_and_prop + 1):
            pval -= b.pmf(k)

    cache[(Ti,Pcj,Nij)] = pval
    cache[(Ti,Pcj)] = b
    return pval


class binocache():
    def __init__(self):
        self.cach = dict()
    def __getitem__(self, item):
        return self.cach.get(item)
    def __setitem__(self, key, value):
        self.cach[key] = value


def best_broaders_enrichment_suprise(supers_for_all_entities: Dict,
                                     per_candidate_links_and_supers: List[Dict],
                                     num_best: int = 5,
                                     super_counts_field: str = "broader_counts",
                                     doprint=False,
                                     representativeness_threshold=0.1,
                                     pvalthrs=0.1, bcach=None):
    Pjs = dict()
    for ent, bros in supers_for_all_entities.items():
        for cat_j in bros:
            Pjs[cat_j] = Pjs.get(cat_j, 0) + 1
    spj = float(sum(list(Pjs.values())))
    Pjs = {c_j: Pjs[c_j] / spj for c_j in Pjs.keys()}
    meanPsize = float(np.mean(list(Pjs.values())))

    sort_by_boringness = lambda x: Pjs[x]

    if bcach is None:
        print("please provide a cache")
        bcach = binocache()

    onlytopmost = []
    for can_i, can in enumerate(per_candidate_links_and_supers):
        Ti = len(can["entities"])
        Nijs = can[super_counts_field]

        # Pjs[c_j] says what portion of the entities in the corpus are tagged with broader c_j
        # Nijs[c_j] says how many entitites of this, the i'th candidate, are in broader c_j
        # good_catjs is the list of broaders which
        #     i) are over-represented in this candidate
        #    ii) make up at least representativeness_threshold of all the entities in this candidate
        good_catjs = {j for j in Nijs.keys()
                      if Nijs[j] > representativeness_threshold * Ti
                      and float(Nijs[j]) / Ti > Pjs[j]}

        if len(good_catjs)==0:
            can["best_match_broader"] = None
            onlytopmost.append(None)
            can["log_odds"] = []
            continue

        pvals = {j: pval(Ti, Pjs[j], Nijs[j], bcach) for j in good_catjs}

        pvalthrs_ = pvalthrs/len(good_catjs)
        significant_over = [j for j in good_catjs if pvals[j] < pvalthrs_]
        insignificant = [j for j in good_catjs if pvals[j] >= pvalthrs_]
        significant_over.sort(key=sort_by_boringness)
        insignificant.sort(key=sort_by_boringness)

        # For this candidate, cats_by_suprise is the list of broaders, sorted by how rare they are in the corpus
        # the statistically over-represented come first, then those which are over-represented although not
        # insignificantly.  Those which aren't over-represented are ignored
        cats_by_suprise = significant_over + insignificant


        maxbroads = min(len(cats_by_suprise), num_best)
        over_represented_rarest_first = []
        for bj in range(maxbroads):
            thisbroad = cats_by_suprise[bj]
            over_represented_rarest_first.append({"candidatesbroader": thisbroad,
                                                  "loggods": pvals[thisbroad],
                                                  "suprise":Pjs[thisbroad]/meanPsize})
        can["log_odds"] = over_represented_rarest_first

        if doprint:
            print("\t\t---", ", ".join([str(pvals[xj]) for xj in over_represented_rarest_first[:maxbroads]]))
        if len(cats_by_suprise)>0:
            can["best_match_broader"] = cats_by_suprise[0]
            onlytopmost.append(pvals[cats_by_suprise[0]])
        else:
            can["best_match_broader"] = None
            onlytopmost.append(None)

    return onlytopmost


def best_broaders(supers_for_all_entities: Dict,
                  per_candidate_links_and_supers: List[Dict],
                  num_best: int = 5,
                  super_counts_field: str = "broader_counts",
                  doprint=False,
                  representativeness_threshold=0.1):
    """
    Returns the best matching super for a candidate class, according to a list of supers for entities in the class
    and entities in the whole corpus. If comparing to a taxonomy, a super is a broader.

    @param super_counts_field:
    @param super_counts: a dictionary that has, for every possible entity, the supers it belongs to
    @param per_candidate_links_and_supers:  a list of dictionaries, one per candidate. Fro each, at least
    two fields are expected "entities" containing the list of entities, and that given by super_counts_field
    which is, in turn, a dictionary whose keys are supers and whose values are the number of entities in that
    candidate having this broad
    @param num_best: maximum number of best matching supers to be returned
    @return: for every candidate class, the num_best best matching supers and their log odds ratio
    """
    result = []
    global_counts = dict()
    for ent, bros in supers_for_all_entities.items():
        for bro in bros:
            global_counts[bro] = global_counts.get(bro, 0) + 1

    onlytopmost = []
    for can in per_candidate_links_and_supers:

        # For this entity, the following dictionaries have an element for every possible super
        # Using notation from the paper
        # T_cc  : The number of entities narrower to a candidate which are tagged with NER typeT
        T_cc = {x: y for x, y in can[super_counts_field].items()
                if y > representativeness_threshold * len(can["entities"])}
        if len(T_cc) == 0:
            T_cc = {x: y for x, y in can[super_counts_field].items()}
        # T_w  :  is the number of entities in the wholecorpus tagged with  T
        T_w = {y: global_counts[y] for y in T_cc.keys()}
        # w : the total number of entities in the whole corpus
        w = float(len(supers_for_all_entities))
        # cc : the total number of entities in this candidate
        cc = float(len(can["entities"]))

        # dict of the form  super : log_odds
        log_odds_per_super = {x: math.log((T_cc[x] / cc) / (T_w[x] / w))
                              for x in T_cc.keys()}

        logslist = list(log_odds_per_super.items())
        logslist.sort(key=lambda x: x[1])
        logslist.reverse()

        maxbroads = min(len(logslist), num_best)
        logodds = []
        for bi in range(maxbroads):
            logodds.append({"candidatesbroader": logslist[bi][0],
                            "loggods": logslist[bi][1]})
        can["log_odds"] = logodds
        if doprint:
            print("\t\t---", ", ".join([str(x[1]) for x in logslist[:maxbroads]]))
        if len(logslist) > 0:
            onlytopmost.append(logslist[0][1])
            can["best_match_broader"] = logslist[0][0]
        else:
            onlytopmost.append(None)
            can["best_match_broader"] = None

    return onlytopmost
