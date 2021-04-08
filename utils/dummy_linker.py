from utils.entity_linker import EntityLinker
import uuid
import logging


class DummyLinker(EntityLinker):

    def __init__(self,
                 language: str = 'en'):
        logging.info("Dummy Linker initialized with language ", language)
        self.cache = dict()
        self.language = language

    def _generate_uri(self, localname=""):
        ln = "_".join(localname.lower().split())
        return "<https://some.uri/" + str(uuid.uuid4()) + "/" + ln + ">"

    def link_within_context(self,
                            surface_form: str,
                            start_offset: int,
                            end_offset: int,
                            context: str):
        key_ = self._gen_key(surface_form=surface_form,
                             context=context)
        uri = self.cache.get(key_, self._generate_uri(surface_form))
        self.cache[key_] = uri
        return uri

    def link_standalone(self,
                        surface_form: str):
        key_ = self._gen_key(surface_form=surface_form)
        uri = self.cache.get(key_, self._generate_uri(surface_form))
        self.cache[key_] = uri
        return uri
