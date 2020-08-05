from typing import Dict, List

import fca


def get_cxt(ners2preds: Dict[str, List[str]]) -> fca.Context:
    objects = list(ners2preds.keys())
    attributes = list(set(sum(ners2preds.values(), [])))
    table = [[att in ners2preds[obj] for att in attributes] for obj in objects]
    cxt = fca.Context(objects=objects, attributes=attributes, cross_table=table)
    return cxt


def clean_cxt(cxt: fca.Context,
              min_att_extent: int = None,
              min_att_len: int = None) -> fca.Context:
    if (min_att_extent and min_att_extent > 1) or (min_att_len):
        to_del = []
        for att in cxt.attributes:
            if ((min_att_extent and len(cxt.aprime([att])) < min_att_extent) or
                    (min_att_len and len(att) < min_att_len)):
                to_del.append(att)
        cxt.delete_attributes(to_del)
    return cxt
