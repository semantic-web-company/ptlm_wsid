from pathlib import Path
import argparse
import configparser
import logging, logging.config
from typing import List, Dict
import re
import random as ran
import os.path as op

import scipy.stats
from scipy import stats
import numpy as np
import json
import functools

parser = argparse.ArgumentParser(description='Type induction on WikiNer corpus.')
parser.add_argument('config_path',
                    # metavar='N', nargs='+',
                    type=str,
                    help='Relative (to wikiner.py script) path to the config file')
args = parser.parse_args()

this_dir = Path(__file__).parent
config_paths = [this_dir / args.config_path, this_dir / 'configs/logging.conf']
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(config_paths)
logger = logging.getLogger()

def kl_vs_random(log_odds_of_induced,
                 randomized_logodds,
                 numbins : int = 300):
    odds_random = [item for sublist in randomized_logodds for item in sublist]
    kde_random = stats.gaussian_kde(odds_random)
    try:
        kde_predicted = stats.gaussian_kde(log_odds_of_induced)

        minpoint = min(odds_random+log_odds_of_induced)
        maxpoint = max(odds_random+log_odds_of_induced)
        supportwidth = maxpoint-minpoint
        minpoint -= 0.5*supportwidth
        maxpoint += 0.5*supportwidth
        binwidth = float(maxpoint-minpoint)/float(numbins)
        bins = [minpoint+i*binwidth for i in range(numbins+1)]
        pdf_random = kde_random(bins)
        pdf_predicted = kde_predicted(bins)

        kl = stats.entropy(pk=pdf_random, qk=pdf_predicted)
        return kl
    except:
        return 0


def oddsratios_probs_vs_random(log_odds_of_induced,
                               randomized_logodds):
    odds_random = [item for sublist in randomized_logodds for item in sublist]
    minpoint = min(odds_random+odds_random)
    maxpoint = max(odds_random+odds_random)
    supportwidth = maxpoint-minpoint
    minpoint -= supportwidth
    maxpoint += supportwidth

    kernel = stats.gaussian_kde(odds_random)
    probs = [1 - kernel.integrate_box_1d(minpoint, loid) for loid in log_odds_of_induced]
    #print("\n")
    #print("loginduced", log_odds_of_induced)
    #print("probs: ", probs)
    pprod = functools.reduce(lambda x, y: x * y, probs)
    #print("\t>", np.mean(probs), pprod, max(probs), sep="\t")
    return np.mean(probs)


def oddsratios_from_mean_of_random(log_odds_of_induced,
                                   randomized_logodds,
                                   dev_thrs=2):
    odds_random = [item for sublist in randomized_logodds for item in sublist]
    meanrandom = np.mean(odds_random)
    stdrandom = np.std(odds_random)
    return len([x for x in log_odds_of_induced
            if (x-meanrandom) > dev_thrs*stdrandom])


def load_candidates(file_path: str,
                    pattern: str = "k10_th3_type%d.json"):
    """
    @param file_path: directory where the candidate dictionaries is
    @param pattern:   patternof the filenames, e.g. k10_th3_type%d.json"
    @return:          a list of dictionaries, one per candidate. Each dictinoary has,
    at least, the keys "descriptors" and "entities"

    """
    if not op.isdir(file_path):
        logging.error("Path for new_types is invalid")
        raise ValueError
    result = []
    fnum = 0
    while True:
        try:
            fname = pattern % fnum
            with open(op.join(file_path, fname)) as fin:
                jstr = fin.read()
                j = json.loads(jstr)
                result.append(j)
            fnum += 1
        except Exception as e:
            if len(result) == 0:
                logging.error("error loading file " + str(fname) + "\n\t" + str(e))
                logging.exception(str(e))
            break
    logging.info("Loaded up to file ", pattern % (fnum - 1),
                 "in total ", len(result), "new_types")
    return result


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
        totfails = 12
        finished = False
        while not finished and totfails>0:
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
                        totfails -= 1
                        break
                all_entities = all_entities[siz:]
                di = {"entities": ents,
                      "descriptors": ["replicate_" + str(nr),
                                      "candidate_" + str(sn)]}
                this_replicate.append(di)
            result.append(this_replicate)
    return result


def get_params_from_dirname(dirname: str,
                            parnames: List[str] = ["k", "th", "m"]):
    if dirname.endswith("/"):
        comptokens = dirname.split("/")[-2]
        mtokens = dirname.split("/")[-2]
    else:
        comptokens = dirname.split("/")[-1]
        mtokens = dirname.split("/")[-2]

    comptokens = comptokens + "_" + mtokens
    pardict = {}
    parlist = comptokens.split("_")
    for pp in parlist:
        for pn in parnames:
            if pp.startswith(pn):
                valstr = pp[len(pn):]
                if "." in valstr:
                    val = float(valstr)
                else:
                    val = int(valstr)
                pardict[pn] = val
                break

    return pardict


def collect_entities(candidates: List[Dict]):
    allentities = set()
    for candidate in candidates:
        allentities = allentities | set(candidate["entities"])

    return allentities
