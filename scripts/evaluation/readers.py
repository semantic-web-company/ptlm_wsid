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
            with open(op.join(file_path,fname)) as fin:
                jstr = fin.read()
                j = json.loads(jstr)
                result.append(j)
            fnum +=1
        except Exception as e:
            logging.error("error loading file "+str(fname)+"\n\t"+str(e))
            if len(result)==0:
                logging.exception(str(e))
            break
    logging.info("Loaded up to file ",pattern%(fnum-1),
                 "in total ",len(result), "candidates")
    return result

