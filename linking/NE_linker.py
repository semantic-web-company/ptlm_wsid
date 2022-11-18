import configparser
import csv
from pathlib import Path
import linking.utils as utils

from linking.entity_linker import EntityLinker


class NELinker(EntityLinker):
    def __init__(self,
                 config_path: str = Path(__file__).parent.parent / "scripts/configs/wikiner_en_senses_config.conf"):
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path)
        nes_in_cxts_path = Path(config['wikiner']['ner_contexts_output'])
        with open(nes_in_cxts_path, 'r') as f:
            self.dict_data = []
            for row in f:
                # @Anna, I have removed the use of the CSV reader as this was not reading all the rows in the
                # tsv file for some reason... will debug later :)

                line = row.strip().split("\t")
                d = {"uri": line[-1].strip(),
                     "surface_form": line[0].split('::')[0],
                     "type": line[0].split('::')[-1],
                     "start_offset": line[1],
                     "end_offset": line[2],
                     "context": line[3]
                     }
                self.dict_data.append(d)
        self.uri_dict = None
        self.surface_dict = None

    def find_label(self, uri:str) -> str:
        return uri

    def link_within_context(self,
                            surface_form: str,
                            start_offset: int,
                            end_offset: int,
                            context: str) -> str:
        if self.surface_dict is None:
            self.surface_dict = {}
            for d in self.dict_data:
                utils.add_or_append(self.surface_dict, key=d['surface_form'], value=d)
        if surface_form in self.surface_dict.keys():
            try:
                idx = self.surface_dict[surface_form]['context'].index(context)
                return self.surface_dict[surface_form]['type'][idx]
            except ValueError:
                raise ValueError(f'context \'{context}\' was not found in the dataset')
        else:
            raise ValueError(f'no contexts for surface form \'{surface_form}\' were not found in the dataset')

    def find_broaders(self, uri: str) -> list:
        uri = uri.split("##")[0]
        if self.uri_dict is None:
            self.uri_dict = {}
            print("initializing")
            for d in self.dict_data:
                # @Anna, I have modified this and am not using your dict utility functions any more
                # I am not sure if the functionality is the same as you intended
                if d['uri'] != '<https://no.entity.found>':
                    keytouse = d['uri']
                else:
                    keytouse = d["surface_form"]
                l = self.uri_dict.get(keytouse, [])
                l.append(d["type"])
                self.uri_dict[keytouse] = l
            for k, v in self.uri_dict.items():
                self.uri_dict[k] = list(set(v))
        if uri in self.uri_dict.keys():
            return self.uri_dict[uri]
        else:
            raise ValueError(f'uri \'{uri}\' was not found in the dataset')
