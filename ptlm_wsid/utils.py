from typing import Dict, Tuple, Iterable

import fca


def get_clean_cxt(ners2preds: Dict[str, Iterable[str]],
                  min_att_extent=3) -> fca.Context:
    objects = list(ners2preds.keys())
    attributes = list(set(sum(ners2preds.values(), [])))
    table = [[att in ners2preds[obj] for att in attributes] for obj in objects]
    cxt = fca.Context(objects=objects, attributes=attributes, cross_table=table)
    #
    to_del = []
    for att in cxt.attributes:
        if len(cxt.get_attribute_extent(att)) < min_att_extent:
            to_del.append(att)
    cxt.delete_attributes(to_del)
    return cxt
