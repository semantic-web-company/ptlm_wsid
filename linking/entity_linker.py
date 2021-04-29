from abc import ABC
import logging


class EntityLinker(ABC):
    """
    Abstract class for an entity linked
    """

    def _gen_key(self,
                 surface_form: str,
                 context: str = ""):
        return str(surface_form.__hash__())+"_@_"+str(context.__hash__())

    def link_within_context(self,
                            surface_form: str,
                            start_offset: int,
                            end_offset: int,
                            context: str) -> str:
        pass

    # def link_standalone(self,
    #                     surface_form: str):
    #     pass

    def find_broaders(self, uri: str) -> list:
        pass

    def find_label(self, uri:str) -> str:
        pass

    def doc(self):
        """
        Should print and return a string explaining additional steps necessary to
        use this linker.
        """
        pass
