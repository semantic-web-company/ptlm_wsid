from linking.entity_linker import EntityLinker
import uuid
import logging
from linking.utils import is_uri

class DummyLinker(EntityLinker):

    def __init__(self,
                 language: str = 'en'):
        logging.info("Dummy Linker initialized with language ", language)
        self.cache = dict()
        self.language = language

    def _generate_uri(self, localname=""):
        ln = "_".join(localname.lower().split())
        # return "<https://some.uri/" + str(uuid.uuid4()) + "/" + ln + ">"
        return f"<{ln}>"

    def find_broaders(self, uri: str):
        if is_uri(uri):
            if "#" in uri:
                ln = uri.split("#")[-1]
            else:
                ln = uri.split("/")[-1]
        else:
            ln = "_".join(uri.split())
        numbroaders = min([5, len(ln)])
        broaders = ["<https://broaders.url/" + ln[i].upper()+str(i) + ">"
                    for i in range(numbroaders)]

        return broaders

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

    def doc(self):
        s = """
        This is a dummy implementation of an Entity Linker
        It will link each surface form with a dummy entity.
        It completely disregards the contexts of the entities.
        Use only for testing.
        
        """