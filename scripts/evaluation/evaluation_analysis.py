import os.path as op
import json
import itertools
import numpy as np
import copy

from matplotlib import pylab as P
from matplotlib import cm as cmaps
import matplotlib.pyplot as plt

from utils import config, load_candidates, config_paths
from broader_analysis import link_and_find_broaders, link_and_find_all_broaders, best_broaders, add_labels_to_supers
from linking.wikidata_linker import WikidataLinker
from linking.NE_linker import NELinker
from scripts.evaluation.plotting import plot2dexploration

evaluation_output_directory = config['evaluation']['outputfolder']
induction_result_directory = config['experiment']['new_types_output_folder']
language = config['wikiner']['language']

# Which are the quantitative eval columns
## FIRST FIGURE:
col_crit = [11,13]

## SECOND FIGURE:
col_crit = [12]



#   11 propr sig enrich
#   12 propr small
#   13 max pval

# col_crit = [7, 8, 9, 10]  # Only sizes
col_params = [0, 1, 2]  # Which are the parameters of the grid search
col_modalities = [3]  # The value of this colum groups comparable measurements
supergrid_col = 2
count_col = 7
multicritcols = [4, 5, 6]  # These criteria (quants) will be  plotted for each modality

figsizex = 388 / 96.9
figsizey = 818 / 96
if len(col_crit)==1:
    figsizey = 600 / 96
    figsizex = 600 / 96.9
outname = "pvals_"+language+".png" if len(col_crit)>1 else "sizes_"+language+".png"

paramvals = {}
modalvals = {}

min_sizes = {"en": 10, "de": 18}
evaluation_csv_file = op.join(str(evaluation_output_directory)[:-1],
                              language + "_" + str(min_sizes[language]) + "_results.csv")

crit_translator = {'proportion_significantly_enriched_HIB': "Proportion of significantly\nenriched candidate topics",
                   'max_pvalue': "Maximum p-value of a topic\nbeing enriched for its best matching category",
                   "proportion_matching_smaller_than_average_broaders_HIB": "Proportion of candidates matching\ncategories smaller than the mean"}

mod_translator = {'NELinker': "Orginal NER annotations", 'WikidataLinker':"Wikidata categories"}

def trans_val(coln: int, val: str):
    """
    How to cast the strings that are in the CSV
    @param coln:   columnnumber
    @param val:  value as appears in the CSV
    @return:
    """
    if coln in col_crit + multicritcols:
        return float(val)
    if coln in col_params + [count_col]:
        return int(val)
    return val


def exclude(d):
    if d["th"] in [5, 6] and d["k"] == 10:
        return True
    return False

fig = plt.figure( figsize=(figsizex, figsizey))
subfigs = fig.subfigures(2, len(col_crit), wspace=0, hspace=0)

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

    # Post-processing on the data
    for d in data:
        try:
            d['oddsratios_probs_vs_random_LIB'] = 1 - d['oddsratios_probs_vs_random_LIB']
            d['above2std_of_oddsrations_vs_random_HIB'] = (float(d['above2std_of_oddsrations_vs_random_HIB']) /
                                                           d['number_of_candidates'])
        except:
            break

    criteria = [htrans[i] for i in col_crit]

    crit_ordering = [max for c in criteria]

    quantmines = {cri: min([dc[cri] for dc in data if not exclude(dc)]) for cri in criteria}
    quantmaxes = {cri: max([dc[cri] for dc in data if dc[cri] < np.inf and not exclude(dc)])
                  for cri in criteria}
    if 'KL_vs_random_HIB' in quantmaxes.keys() and quantmaxes['KL_vs_random_HIB'] > 20:
        cri = 'KL_vs_random_HIB'
        quantmaxes[cri] = max([dc[cri] for dc in data if dc[cri] < 20])

    quantmaxes['max_pvalue']=0.6
    quantmines['max_pvalue']=0
    quantmaxes['proportion_significantly_enriched_HIB'] = 1
    quantmines['proportion_significantly_enriched_HIB'] = 0.76

    subfigurenum = -1
    for mt in col_modalities:
        lmodals = list(modalvals[htrans[mt]])
        lmodals.sort()
        lmodals.reverse()
        for modnum,mod in enumerate(lmodals):  # Figure   Linker
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

            muliplotcrit = [htrans[x] for x in multicritcols]
            for crinum, cri in enumerate(criteria):  # Figure    quant
                if mod=="NELinker" and cri in ["proportion_matching_smaller_than_average_broaders_HIB"]:
                    continue
                quantmax = quantmaxes[cri]
                quantmin = quantmines[cri]
                cm = cmaps.get_cmap('viridis', 64)
                cm.set_under([1, 1, 1, 1])

                # Here we create a new subfigure

                if len(col_crit)>1:
                    sf = subfigs[modnum,crinum]
                    sfsubplots = sf.subplots(2, 2, sharex=True, sharey=True)
                else:
                    sf = subfigs[modnum]
                    sfsubplots = sf.subplots(1, 4, sharex=True, sharey=True)


                subplotn = 0
                for supergridval in paramvals[htrans[supergrid_col]]:  # subplot  th
                    subplotn += 1
                    if len(col_crit)>1:
                        subplotrow = 0 if subplotn in [1,2] else 1
                        subplitcol = 0 if subplotn in [1,3] else 1
                        ax = sfsubplots[subplotrow, subplitcol]
                    else:
                        ax = sfsubplots[subplotn-1]
                        subplotrow = 0
                        subplitcol = subplotn-1

                    mtitle = mod if cri in muliplotcrit else ""
                    gridtitle = mtitle + "  " + htrans[supergrid_col] + "=" + str(supergridval)


                    # Here we create a new subplot
                    pc = plot2dexploration(mod, cri, supergridval,
                      paramvals, htrans, col_params, supergrid_col,
                      gridcols, gridaxes, data_c, mt,
                      exclude, quantmax, quantmin,
                      ax, cm)

                    if subplotrow==0:
                        ax.set_xlabel("")
                    if subplitcol>0:
                        ax.set_ylabel("")

                if len(col_crit) == 1:
                    wsp=0.12
                    cbarpos = 0.25

                else:
                    wsp=0.0
                    cbarpos= 0.05
                if crinum==0 and len(col_crit) > 1:
                    textypos = 0.25 if modnum == 0 else 0.35
                    sf.text(x=0.01, y = textypos, s=mod_translator[mod],
                            fontsize="x-large", rotation="vertical")
                if modnum==0:
                    sf.suptitle(crit_translator.get(cri, cri), fontsize="x-large")
                    sf.subplots_adjust(left=0.138, right=1, bottom=0.12, top=0.87, wspace=wsp, hspace=0.05)
                    #fig.subplots_adjust(left=0, right=0.911, bottom=0.11, top=0.99, wspace=0.0, hspace=0.11)
                if modnum==1 or len(col_crit)==1:
                    if len(col_crit)==1:
                        toplim = 1
                    else:
                        toplim = 1
                    #fig.subplots_adjust(left=0, right=0.911, bottom=0.31, top=0.99, wspace=0.0, hspace=0.11)
                    sf.subplots_adjust(left=0.138, right=1, bottom=0.22, top=toplim, wspace=wsp, hspace=0.05)
                    cbarax = sf.add_axes([0.2, cbarpos, 0.7, 0.057])
                    #[left, bottom, width, height]
                    sf.colorbar(pc, cax=cbarax,  orientation="horizontal")

                    #P.set_cmap(cm)
                    #P.colorbar()

#P.subplots_adjust(left=0, right=1, bottom=0.22, top=0.88, wspace=0.0, hspace=0.07)

#   Plot number of entities vs each meassure
#   Artem:  en:10  de:20  could be a good threshold

linkers = [
    # DummyLinker(),
    NELinker(config_path=config_paths[0]),
    WikidataLinker(language=language)
]



P.savefig(outname)
#P.show()
