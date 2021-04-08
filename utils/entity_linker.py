from abc import ABC
import logging


class EntityLinker(ABC):

    def __init__(self, config: dict):
        pass

    def _gen_key(self,
                 surface_form: str,
                 context: str = ""):
        return str(surface_form.__hash__())+"_@_"+str(context.__hash__())

    def link_within_context(self,
                            surface_form: str,
                            start_offset: int,
                            end_offset: int,
                            context: str):
        pass

    def link_standalone(self,
                        surface_form: str):
        pass
