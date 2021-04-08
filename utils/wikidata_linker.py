import os.path as op
import uuid
import logging
import pickle
import time
import json

import requests

from utils.entity_linker import EntityLinker


# To run the fishing server
#  docker run --rm --name fishing_ctr -p 8090:8090 -v ${PWD}/data/db:/fishing/nerd/data/db/ -it fishing

placeholder_texts ={"en":" is a concept we want to find",
                    "de":" ist ein Konzept wir finden willen",
                    "es":" es un concepto que queremos encontrar"}

class WikidataLinker(EntityLinker):

    def __init__(self, config=None):
        if config is None:
            config=dict()
        super().__init__(config)
        self.linking_url = config.get("fishing_url", "http://localhost:8090")
        self.sparql_endpoint = config.get("wikidata_sparql","https://query.wikidata.org/sparql")
        self.caches_dir = config.get("linking_caches_directory","/tmp")
        language = config.get("language", "en")

        self.label_cache = dict()
        self.linking_cache = dict()
        self.KG_cache = dict()
        self.label_cache_file = None
        self.linking_cache_file = None
        self.KG_cache_file = None

        if self.caches_dir is not None and not op.exists(self.caches_dir):
            logging.error("Cache dir was supplied for Wikidata linker, but it doesn't exist")
            logging.error("Will revert to no-cache version")
            self.caches_dir = None

        if self.caches_dir is not None:
            self.label_cache_file = op.join(self.caches_dir,"Wikidata_"+language+"_labelcache.pkl")
            self.linking_cache_file = op.join(self.caches_dir,"Wikidata_"+language+"_linkingcache.pkl")
            self.KG_cache_file = op.join(self.caches_dir, "Wikidata_" + language + "_KGcache.pkl")

            if op.isfile(self.label_cache_file):
                with open(self.label_cache_file,"rb") as fin:
                    self.label_cache = pickle.load(fin)

            if op.isfile(self.linking_cache_file):
                with open(self.linking_cache_file, "rb") as fin:
                    self.linking_cache = pickle.load(fin)

            if op.isfile(self.KG_cache_file):
                with open(self.KG_cache_file, "rb") as fin:
                    self.KG_cache = pickle.load(fin)

        self.language = language
        logging.info("Wikidata Linker initialized with language ", language)

    def write_cache(self):
        if self.caches_dir is not None:
            with open(self.KG_cache_file, "wb") as fout:
                pickle.dump(self.KG_cache, fout)
            with open(self.linking_cache_file, "wb") as fout:
                pickle.dump(self.linking_cache, fout)
            with open(self.label_cache_file, "wb") as fout:
                pickle.dump(self.label_cache, fout)

    def clear_cache(self):
        self.label_cache = dict()
        self.linking_cache = dict()
        self.KG_cache = dict()

    def _query_fishing_kb(self, concept, max_retries=2):
        if concept in self.KG_cache.keys():
            return self.KG_cache[concept]
        lang = self.language
        retries = 0
        url = self.linking_url + "/service/kb/concept/" + str(concept) + "?lang=" + lang
        response = None
        while response is None and retries < max_retries:
            try:
                response = requests.request("GET", url)
            except:
                time.sleep(0.5 * retries + 0.01)
        if response is None:
            print("Failed for ", concept, end=".  ")
            response = []
        self.KG_cache[concept] = response
        return response


    def query_fishing_disamb(self, text, max_retries=2):
        lang = self.language
        url = self.linking_url + "/service/disambiguate"
        strict_size = None

        payload = { "language": {"lang" : lang }}
        if len(text) > 6:
            payload["text"] = text
        else:
            payload["shortText"] = text

        headers = {'Content-Type': 'application/json; charset=utf-8'}
        response = None
        retries = 0

        while response is None and retries < max_retries:
            retries += 1
            try:
                jst = json.dumps(payload, indent=2, ensure_ascii=False)
                response = requests.request("POST", url, headers=headers, data=jst.encode("utf-8"))
            except Exception:
                time.sleep(0.015 * retries + 0.051)
        if response is None:
            print("No response for ", text)
            return None
        # print("got:   ",response.text)

        return response.json()



    def _generate_uri(self, localname=""):
        ln = "_".join(localname.lower().split())
        return "<https://some.uri/" + str(uuid.uuid4()) + "/" + ln + ">"

    def link_within_context(self,
                            surface_form: str,
                            start_offset: int,
                            end_offset: int,
                            context: str,
                            minjaccard:int = 0.5,
                            minscore:int = 0.2):
        key_ = self._gen_key(surface_form=surface_form,
                             context=context)
        if key_ in self.linking_cache:
            return self.linking_cache[key_]

        matches = []
        resp = self.query_fishing_disamb(text=context)
        if "entities" in resp.keys():
            matches = resp["entities"]

        logging.debug(str(len(matches))+" matches were found!"+"\n\t"+"\n\t".join([e["rawName"] for e in matches]))
        maxjaccard = minjaccard
        bestscore = minscore
        bestmatching = "<https://no.entity.found>"
        for ent in matches:
            logging.debug(json.dumps(ent,indent=2))
            if any([x not in ent.keys() for x in ["wikidataId", "offsetStart", "offsetEnd"]]):
                logging.debug("\t skipping for no id" )
                continue
            if ent["offsetStart"] > end_offset or ent["offsetEnd"] < start_offset:
                logging.debug("\tskipping for no intersection "+ent.get("rawName","_"))
                continue

            # the entity has to overlap with the surface_form in some percentage. here we compute it.
            inter_start = max([ent["offsetStart"], start_offset])
            inter_end = min([ent["offsetEnd"], end_offset])
            union_start = min([ent["offsetStart"], start_offset])
            union_end = max([ent["offsetEnd"], end_offset])
            jaccard = float(inter_end-inter_start)/(union_end-union_start)

            # if the entity has no scores, we take it
            # if it has some scores, we take it only if the best of them is not too bad
            scores = []
            if "nerd_selection_score" in ent.keys():
                scores.append(float(ent["nerd_selection_score"]))
            if "nerd_score" in ent.keys():
                scores.append(float(ent["nerd_score"]))
            if len(scores)==0:
                score = minscore + 0.01
            else:
                score = max(scores)

            if score>bestscore*0.8 and jaccard > maxjaccard:
                bestmatching = "<http://www.wikidata.org/entity/"+ent["wikidataId"]+">"
                maxjaccard = jaccard
            else:
                logging.debug("\tnot good enough "+str(jaccard)+"_"+str(score))


        self.linking_cache[key_] = bestmatching
        return bestmatching


    def link_standalone(self,
                        surface_form: str):

        artificial_context = surface_form + placeholder_texts.get(self.language,
                                                                  placeholder_texts["en"])

        return self.link_within_context(surface_form=surface_form,
                                        context=artificial_context,
                                        start_offset=0,
                                        end_offset=len(surface_form))
