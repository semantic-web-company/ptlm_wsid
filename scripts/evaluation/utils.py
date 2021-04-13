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



def collect_entities(candidates: List[Dict]):
    allentities = set()
    for candidate in candidates:
        allentities = allentities | set(candidate["entities"])

    return allentities

