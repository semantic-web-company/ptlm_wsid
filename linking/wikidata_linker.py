import os.path as op
import uuid
import logging
import pickle
import time
import json

import requests

from linking.entity_linker import EntityLinker

placeholder_texts = {"en": " is a concept we want to find",
                     "de": " ist ein Begriff, den wir finden wollen",
                     "es": " es un concepto que queremos encontrar"}


class WikidataLinker(EntityLinker):
    def __init__(self,
                 el_url: str = "http://localhost:8090",
                 wikidata_sparql: str = "https://query.wikidata.org/sparql",
                 linking_caches_directory: str = "/tmp",
                 language: str = "en",
                 cache_writes_before_persist: int = 200,
                 ):

        self.linking_url = el_url
        self.sparql_endpoint = wikidata_sparql
        self.caches_dir = linking_caches_directory

        self.label_cache = dict()
        self.linking_cache = dict()
        self.KG_cache = dict()
        self.label_cache_file = None
        self.linking_cache_file = None
        self.KG_cache_file = None

        self.KGcache_misses = 0
        self.label_cache_misses = 0
        self.linking_cache_misses = 0
        self.cache_misses_before_persist = 0

        if self.caches_dir is not None and not op.exists(self.caches_dir):
            logging.error("Cache dir was supplied for Wikidata linker, but it doesn't exist")
            logging.error("Will revert to no-cache version")
            self.caches_dir = None

        if self.caches_dir is not None:
            self.label_cache_file = op.join(self.caches_dir, "Wikidata_" + language + "_labelcache.pkl")
            self.linking_cache_file = op.join(self.caches_dir, "Wikidata_" + language + "_linkingcache.pkl")
            self.KG_cache_file = op.join(self.caches_dir, "Wikidata_" + language + "_KGcache.pkl")

            if op.isfile(self.label_cache_file):
                with open(self.label_cache_file, "rb") as fin:
                    self.label_cache = pickle.load(fin)

            if op.isfile(self.linking_cache_file):
                with open(self.linking_cache_file, "rb") as fin:
                    self.linking_cache = pickle.load(fin)

            if op.isfile(self.KG_cache_file):
                with open(self.KG_cache_file, "rb") as fin:
                    self.KG_cache = pickle.load(fin)

        self.language = language
        if not self._test():
            logging.error("Wikidata Linker initialization failed")
            logging.info(self.doc())
        logging.info("Wikidata Linker initialized with language ", language)

    def _write_cache(self, onlyExpired=False):
        if self.caches_dir is None:
            return None
        writeKG = writeLinking = writeLabel = False
        if onlyExpired:
            if self.linking_cache_misses > self.cache_misses_before_persist:
                writeLinking = True
                self.linking_cache_misses = 0
            if self.label_cache_misses > self.cache_misses_before_persist:
                writeLabel = True
                self.label_cache_misses = 0
            if self.KGcache_misses > self.cache_misses_before_persist:
                writeKG = True
                self.KGcache_misses = 0
        else:
            writeKG = writeLinking = writeLabel = False

        if writeKG:
            with open(self.KG_cache_file, "wb") as fout:
                pickle.dump(self.KG_cache, fout)
        if writeLinking:
            with open(self.linking_cache_file, "wb") as fout:
                pickle.dump(self.linking_cache, fout)
        if writeLabel:
            with open(self.label_cache_file, "wb") as fout:
                pickle.dump(self.label_cache, fout)

    def _clear_cache(self):
        self.label_cache = dict()
        self.linking_cache = dict()
        self.KG_cache = dict()

    def _get_categories(self, URI, lang="de"):
        if (type(URI) is dict) and ('wikidataId' in URI.keys()):
            URI = URI["wikidataId"]
        r = self._query_fishing_kb(URI, lang=lang)
        j = json.loads(r.text.encode('utf8'))
        results = dict()
        if 'statements' not in j.keys():
            return []
        for cat in [s for s in j['statements'] if s['propertyId'] == "P31"]:
            if "valueName" in cat.keys():
                name = cat["valueName"]
            else:
                name = self._get_entity_label(wid=cat["value"], lang=lang)
            results[cat["value"]] = name
            # results.add({"wikidataId":cat["value"], "name":cat['valueName']})
        resultset = [{"wikidataId": k, "name": v} for k, v in results.items()]
        self._write_cache(onlyExpired=True)

        return resultset

    def _query_fishing_kb(self, concept, max_retries=2, lang="de"):
        if concept in self.KG_cache.keys():
            return self.KG_cache[concept]
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
        self.KGcache_misses += 1
        return response

    def _query_fishing_disamb(self, text, max_retries=2):
        lang = self.language
        url = self.linking_url + "/service/disambiguate"
        strict_size = None

        payload = {"language": {"lang": lang}}
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

    def _get_entity_label(self, wid, lang):
        if (wid, lang) in self.label_cache.keys():
            return self.label_cache[(wid, lang)]
        url = "http://www.wikidata.org/entity/" + wid
        headers = {
            'Accept': 'application/json'
        }
        try:
            response = requests.request("GET", url, headers=headers)
            rj = response.json()
            ents = rj["entities"]
            entks = list(ents.keys())
            entk = wid if wid in entks else entks[0]
            labs = ents[entk]["labels"]
            for l, ld in labs.items():
                if l[:2] == lang:
                    self.label_cache[(wid, lang)] = ld["value"]
                    return ld["value"]
        except:
            print("\t", wid, ""'s', lang, "label not found in wikidata")
        self.label_cache[(wid, lang)] = wid
        self.label_cache_misses += 1


        return wid

    def link_within_context(self,
                            surface_form: str,
                            start_offset: int,
                            end_offset: int,
                            context: str,
                            minjaccard: int = 0.5,
                            minscore: int = 0.2):
        key_ = self._gen_key(surface_form=surface_form,
                             context=context)
        if key_ in self.linking_cache:
            return self.linking_cache[key_]

        matches = []
        resp = self._query_fishing_disamb(text=context)
        if "entities" in resp.keys():
            matches = resp["entities"]

        logging.debug(
            str(len(matches)) + " matches were found!" + "\n\t" + "\n\t".join([e["rawName"] for e in matches]))
        maxjaccard = minjaccard
        bestscore = minscore
        bestmatching = f"<{surface_form}>"
        for ent in matches:
            logging.debug(json.dumps(ent, indent=2))
            if any([x not in ent.keys() for x in ["wikidataId", "offsetStart", "offsetEnd"]]):
                logging.debug("\t skipping for no id")
                continue
            if ent["offsetStart"] > end_offset or ent["offsetEnd"] < start_offset:
                logging.debug("\tskipping for no intersection " + ent.get("rawName", "_"))
                continue

            # the entity has to overlap with the surface_form in some percentage. here we compute it.
            inter_start = max([ent["offsetStart"], start_offset])
            inter_end = min([ent["offsetEnd"], end_offset])
            union_start = min([ent["offsetStart"], start_offset])
            union_end = max([ent["offsetEnd"], end_offset])
            jaccard = float(inter_end - inter_start) / (union_end - union_start)

            # if the entity has no scores, we take it
            # if it has some scores, we take it only if the best of them is not too bad
            scores = []
            if "nerd_selection_score" in ent.keys():
                scores.append(float(ent["nerd_selection_score"]))
            if "nerd_score" in ent.keys():
                scores.append(float(ent["nerd_score"]))
            if len(scores) == 0:
                score = minscore + 0.01
            else:
                score = max(scores)

            if score > bestscore * 0.8 and jaccard > maxjaccard:
                bestmatching = "<http://www.wikidata.org/entity/" + ent["wikidataId"] + ">"
                maxjaccard = jaccard
            else:
                logging.debug("\tnot good enough " + str(jaccard) + "_" + str(score))

        self.linking_cache[key_] = bestmatching
        self.linking_cache_misses += 1
        self._write_cache(onlyExpired=True)
        return bestmatching

    def link_standalone(self,
                        surface_form: str):
        artificial_context = surface_form + placeholder_texts.get(self.language,
                                                                  placeholder_texts["en"])

        return self.link_within_context(surface_form=surface_form,
                                        context=artificial_context,
                                        start_offset=0,
                                        end_offset=len(surface_form))

    def find_broaders(self, uri:str):
        self._get_categories(URI=uri, lang=self.language)

    def _test(self):
        try:
            self.link_standalone("Albert Einstein")
            return True
        except:
            return False

    def doc(self):
        s = """
        This class wraps access to Entity Fishing (https://github.com/kermitt2/entity-fishing)
        To set up entity fishing, do the following:
        0) Make sure you have docker installed, and some 20GB of storage available
        1) Create a new directory. In this directory:
        2)  ` wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-de.zip `
            ` wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-en.zip `
            ` wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.3/linux/db-kb.zip `
            ` unzip db-de.zip `
            ` unzip db-kb.zip `
            ` unzip db-en.zip `
        3) ` docker run --rm --name fishing_ctr -p 8090:8090 -v ${PWD}/db:/fishing/nerd/data/db/ -it syats/fishing:0.6.0 `

        If you change the port number in the last command, or you deploy the container in a different machine
        you need to provide the ` el_url ` argument to the constructor.

        """
        print(s)
        return s
