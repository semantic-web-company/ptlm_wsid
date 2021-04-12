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
logging.config.fileConfig(config)

urlregex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def collect_entities(candidates: List[Dict]):
    allentities = set()
    for candidate in candidates:
        allentities = allentities | candidate["entities"]

    return allentities

def is_uri(uristr: str) -> bool:
    if uristr[0] != "<" or uristr[-1] != ">":
        return False
    return re.match(urlregex, uristr[1:-1]) is not None
