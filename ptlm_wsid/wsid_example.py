import re
from typing import List

import ptlm_wsid.target_context as tc
import ptlm_wsid.generative_factors as gf


def prepare_target_contexts(cxt_strs: List[str],
                            target_word: str,
                            verbose: bool = True) -> List[tc.TargetContext]:
    """
    The function creates a simple regex from the target word and searches this
    pattern in the context strings. If found then the start and end indices are
    used to produce a TargetContext.

    :param cxt_strs: list of context strings
    :param target_word: the target word
    :param verbose: print also individual predictions
    """
    tcs = []
    for cxt_str in cxt_strs:
        re_match = re.search(target_word, cxt_str, re.IGNORECASE)
        if re_match is None:
            raise ValueError(f'In "{cxt_str}" the target '
                             f'"{target_word}" was not found')
        start_ind, end_ind = re_match.start(), re_match.end()
        new_tc = tc.TargetContext(
            context=cxt_str, target_start_end_inds=(start_ind, end_ind))
        if verbose:
            top_predictions = new_tc.get_topn_predictions()
            print(f'Predictions for {target_word} in {cxt_str}: '
                  f'{top_predictions}')
        tcs.append(new_tc)
    return tcs


if __name__ == '__main__':
    # # Uncomment if you want additional logs, like cluster scores, etc.
    # import logging
    #
    # logging.basicConfig(level=logging.DEBUG)

    cxts_dicts = {
        1: "The jaguar's present range extends from Southwestern United States and Mexico in North America, across much of Central America, and south to Paraguay and northern Argentina in South America.",
        2: "Overall, the jaguar is the largest native cat species of the New World and the third largest in the world.",
        3: "Given its historical distribution, the jaguar has featured prominently in the mythology of numerous indigenous American cultures, including those of the Maya and Aztec.",
        4: "The jaguar is a compact and well-muscled animal.",
        5: "Melanistic jaguars are informally known as black panthers, but as with all forms of polymorphism they do not form a separate species.",
        6: "The jaguar uses scrape marks, urine, and feces to mark its territory.",
        7: "The word 'jaguar' is thought to derive from the Tupian word yaguara, meaning 'beast of prey'.",
        8: "Jaguar's business was founded as the Swallow Sidecar Company in 1922, originally making motorcycle sidecars before developing bodies for passenger cars.",
        9: "In 1990 Ford acquired Jaguar Cars and it remained in their ownership, joined in 2000 by Land Rover, till 2008.",
        10: "Two of the proudest moments in Jaguar's long history in motor sport involved winning the Le Mans 24 hours race, firstly in 1951 and again in 1953.",
        11: "He therefore accepted BMC's offer to merge with Jaguar to form British Motor (Holdings) Limited.",
        12: "The Jaguar E-Pace is a compact SUV, officially revealed on 13 July 2017."}

    titles, cxts = list(zip(*cxts_dicts.items()))  # convert to 2 lists
    tcs = prepare_target_contexts(cxt_strs=cxts, target_word='jaguar')
    senses = gf.induce(
        contexts=[tc.context for tc in tcs],
        target_start_end_tuples=[tc.target_start_end_inds for tc in tcs],
        titles=titles,
        target_pos='N',  # we want only nouns
        n_sense_indicators=5,  # how many substitutes for each sense in the output
        top_n_pred=50)  # the number of substitutes for each context
    for i, sense in enumerate(senses):
        print(f'Sense #{i+1}')
        print(f'Sense indicators: {", ".join(str(x) for x in sense.intent)}')
        print(f'Found in contexts: {", ".join(str(x) for x in sense.extent)}')

    sense_indicators = [list(sense.intent) for sense in senses]
    for tc, title in zip(tcs, titles):
        scores = tc.disambiguate(sense_clusters=sense_indicators)
        print(f'For context: "{str(title).upper()}. {tc.context}" '
              f'the sense: {sense_indicators[scores.index(max(scores))]} '
              f'is chosen.')
