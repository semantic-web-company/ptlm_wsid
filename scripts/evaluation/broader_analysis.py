import math
from typing import List, Dict, Set
from linking.entity_linker import EntityLinker
from linking.utils import is_uri
from linking.dummy_linker import DummyLinker as DuLi


def link_and_find_broaders(candidates: List[Dict],
                           linker: EntityLinker = None):
    """

    @param candidates:
    @param linker:
    @return: a list of candidates, with an extra key called "broader_counts" which is a dictionary
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
        broaders = linker.find_broaders(ent=ent)
        result[ent] = broaders

    return result


def best_broaders(all_super_counts: Dict,
                  per_candidate_links_and_supers: List[Dict],
                  num_best: int = 5):
    """

    @param super_counts:
    @param per_candidate_links_and_supers:
    @param num_best:
    @return: for every candidate class, the num_best best matching broaders and their log odds ratio
    """
    result = []
    global_counts = dict()
    for ent, bros in all_super_counts.items():
        for bro in bros:
            global_counts[bro] = global_counts.get(bro, 0) + 1

    for can in per_candidate_links_and_supers:
        counts = can["broader_counts"]
        countslist = list(counts.items())
        # ToDo: Here we must substitute for log odds ratio actual computation
        countslist.sort(key=lambda x: float(x[1]) / global_counts.get(x[0], 1))
        countslist.reverse()
        maxbroads = min(len(countslist), num_best)
        logodds = []
        for bi in range(maxbroads):

            logodds.append({"broader": countslist[bi][0],
                            "loggods": math.log(countslist[bi][1])})

        can["log_odds"] = logodds