from pathlib import Path
import argparse
import configparser
import logging, logging.config
from typing import List, Dict
import re


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


def get_params_from_dirname(dirname: str,
                            parnames: List[str] = ["k", "th", "m"]):
    if dirname.endswith("/"):
        comptokens = dirname.split("/")[-2]
        mtokens = dirname.split("/")[-2]
    else:
        comptokens = dirname.split("/")[-1]
        mtokens = dirname.split("/")[-2]

    comptokens = comptokens+"_"+mtokens
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

