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


def best_broaders(supers_for_all_entities: Dict,
                  per_candidate_links_and_supers: List[Dict],
                  num_best: int = 5,
                  super_counts_field: str = "broader_counts"):
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
        T_cc = can[super_counts_field]
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
        logslist.sort(key= lambda x: x[1])
        logslist.reverse()

        maxbroads = min(len(logslist), num_best)
        logodds = []
        for bi in range(maxbroads):
            logodds.append({"broader": logslist[bi][0],
                            "loggods": logslist[bi][1]})
        can["log_odds"] = logodds
        onlytopmost.append(logslist[0][1])

    return onlytopmost

