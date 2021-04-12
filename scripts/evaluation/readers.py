import json
import os.path as op
import os
import logging


def load_candidates(file_path: str,
                    pattern: str = "k10_th3_type%d.json"):
    """
    @param file_path: directory where the candidate dictionaries is
    @param pattern:   patternof the filenames, e.g. k10_th3_type%d.json"
    @return:          a list of dictionaries, one per candidate. Each dictinoary has,
    at least, the keys "descriptors" and "entities"

    """
    if not op.isdir(file_path):
        logging.error("Path for candidates is invalid")
        raise ValueError
    result = []
    fnum = 0
    while True:
        try:
            fname = pattern % fnum
            with open(fname) as fin:
                jstr = fin.read()
                j = json.load(jstr)
                result.append(j)
        except:
            break
            logging.error("error loading file "+str(fname))
            raise ValueError
    return result

