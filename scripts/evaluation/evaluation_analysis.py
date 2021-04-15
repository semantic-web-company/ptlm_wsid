import os.path as op
import json
import itertools
import numpy as np
from matplotlib import pylab as P

from utils import config

evaluation_output_directory = config['evaluation']['outputfolder']
language = config['wikiner']['language']
col_crit = [4, 5, 6]  # Which are the quantitative eval columns
col_params = [0, 1, 2]  # Which are the parameters of the grid search
col_modalities = [3]  # The value of this colum groups comparable measurements
supergrid_col = 2
count_col = 7

paramvals = {}
modalvals = {}

evaluation_csv_file = op.join(evaluation_output_directory, language + "_results.csv")


def trans_val(coln: int, val: str):
    """
    How to cast the strings that are in the CSV
    @param coln:   columnnumber
    @param val:  value as appears in the CSV
    @return:
    """
    if coln in col_crit:
        return float(val)
    if coln in col_params+[count_col]:
        return int(val)
    return val


with open(evaluation_csv_file) as fin:
    curr_row = 0
    htrans = {}
    data = []
    for row in fin:
        curr_row += 1
        row_s = row.strip().split("\t")
        if len(row_s) < len(col_crit + col_params + col_modalities):
            continue
        if curr_row == 1:
            for coln, head in enumerate(row_s):
                htrans[coln] = head
            for cp in col_params:
                paramname = htrans[cp]
                paramvals[paramname] = set()
            for cm in col_modalities:
                modname = htrans[cm]
                modalvals[modname] = set()
            continue

        rowd = {htrans[coln]: trans_val(coln=coln, val=val)
                for coln, val in enumerate(row_s)}
        for cp in col_params:
            paramname = htrans[cp]
            paramvals[paramname].add(trans_val(cp, rowd[paramname]))
        for cm in col_modalities:
            modname = htrans[cm]
            modalvals[modname].add(trans_val(cm, rowd[modname]))

        data.append(rowd)

    for d in data:
        d['oddsratios_probs_vs_random_LIB'] = 1 - d['oddsratios_probs_vs_random_LIB']
        d['above2std_of_oddsrations_vs_random_HIB'] = (float(d['above2std_of_oddsrations_vs_random_HIB']) /
                                                       d['number_of_candidates'])

    criteria = [htrans[i] for i in col_crit]

    crit_ordering = [max if c.endswith("HIB") else min
                     for c in criteria]

    for mt in col_modalities:
        for mod in modalvals[htrans[mt]]:  # Figure   Linker
            best_per_criteria = dict()
            data_c = [d for d in data if d[htrans[mt]] == mod]
            print("\n----\n", mod)
            for ci, c in enumerate(criteria):
                ordfunc = crit_ordering[ci]
                maxc = ordfunc([d[c] for d in data_c])
                best_per_criteria[c] = [d for d in data_c if d[c] == maxc]
            print(json.dumps(best_per_criteria, indent=2))

            gridaxes = []
            gridcols = [x for x in col_params if x != supergrid_col]
            for gridcol in gridcols:
                axvals = list(paramvals[htrans[gridcol]])
                axvals.sort()
                gridaxes.append(axvals)

            grid = itertools.product(*gridaxes)
            for a in grid:
                print(a)

            for cri in criteria:  # Figure    quant
                quantmax = max([dc[cri] for dc in data_c])
                quantmin = min([dc[cri] for dc in data_c])
                P.figure(mod + "_" + cri)
                subplotn = 0
                for supergridval in paramvals[htrans[supergrid_col]]:  # subplot  th
                    subplotn += 1
                    P.subplot(2, 2, subplotn)
                    P.title(htrans[supergrid_col] + "=" + str(supergridval))
                    gridtitle = mod + "  " + htrans[supergrid_col] + "=" + str(supergridval)
                    axlengths = [len(paramvals[htrans[gridcol]])
                                 for gridcol in [x for x in col_params if x != supergrid_col]]
                    matr = -1*np.ones(tuple(axlengths))
                    for x1 in range(axlengths[0]):  # xaxis  k
                        for x2 in range(axlengths[1]):  # yaxis  m
                            p1val = gridaxes[0][x1]
                            p2val = gridaxes[1][x2]
                            dset = [x for x in data_c
                                    if x[htrans[gridcols[0]]] == p1val
                                    and x[htrans[gridcols[1]]] == p2val
                                    and x[htrans[mt]] == mod
                                    and x[htrans[supergrid_col]] == supergridval]
                            if len(dset) < 1:
                                continue
                            d = dset[0]
                            matr[x1, x2] = d[cri]
                    P.imshow(matr, vmin=quantmin, vmax=quantmax)
                    P.xlabel(htrans[gridcols[0]])
                    P.ylabel(htrans[gridcols[1]])
                    P.xticks([i for i, x in enumerate(gridaxes[0])],
                             [str(x) for x in gridaxes[0]])
                    P.yticks([i for i, x in enumerate(gridaxes[1])],
                             [str(x) for x in gridaxes[1]])
                    cm = P.get_cmap().copy()
                    cm.set_under([1,1,1,1])
                    P.set_cmap(cm)
                    P.subplots_adjust(left=0, right=1, bottom=0.11, top=0.945,
                                      wspace=0.01, hspace=0.417)

P.show()
