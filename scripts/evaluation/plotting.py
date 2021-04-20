from typing import Set,List,Dict
from matplotlib import pylab as P
import numpy as np
from scipy import stats

def plot2dexploration(mod, cri, supergridval,
                      paramvals, htrans, col_params, supergrid_col,
                      gridcols, gridaxes, data_c, mt,
                      exclude, quantmax, quantmin,
                      ax, cm):
    axlengths = [len(paramvals[htrans[gridcol]])
                 for gridcol in [x for x in col_params if x != supergrid_col]]
    matr = -1 * np.ones(tuple(axlengths))
    sizmtr = np.zeros(tuple(axlengths))
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
            if exclude(d):
                continue
            matr[x1, x2] = d[cri]
            sizmtr[x2, x1] = d['number_of_candidates']
    # matr[sizmtr.nonzero()]=-1

    #P.figure(mod + "_" + cri)
    #P.subplot(2, 2, subplotn)
    # ax = P.gca()
    #ax.set_title(htrans[supergrid_col] + "=" + str(supergridval))
    pc = ax.imshow(matr, vmin=quantmin, vmax=quantmax, origin="lower", cmap=cm)
    ax.set_ylabel(htrans[gridcols[0]])
    ax.set_xlabel(htrans[gridcols[1]])
    xticks = [i for i, x in enumerate(gridaxes[1])]
    yticks = [i for i, x in enumerate(gridaxes[0])]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(x) for x in gridaxes[0]])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in gridaxes[1]])

    for ix, tx in enumerate(xticks):
        for iy, ty in enumerate(yticks):
            tex = str(int(sizmtr[ix, iy])) if sizmtr[ix, iy] > 0 else ""
            ax.text(tx, ty, tex, color="white", ha="center", va="center")



    return pc


def plot_against_randomized(log_odds_of_induced_,
                            randomized_logodds,
                            title: str = "Linking",
                            numbins: int = 10,
                            numticks: int = 5,
                            lang: str = "en",
                            colorrandom: str = "orange",
                            colorhist: str = "cornflowerblue",
                            figsufix="",
                            extratext=None):

    P.figure(title+figsufix)
    odds_random = [item for sublist in randomized_logodds for item in sublist]
    odds_random = [o for o in odds_random if o is not None]
    log_odds_of_induced = [o for o in log_odds_of_induced_ if o is not None]
    if len(log_odds_of_induced)*len(odds_random) == 0:
        return
    maxratio = max(log_odds_of_induced + odds_random)
    minratio = min(log_odds_of_induced + odds_random)
    binsize = (maxratio - minratio) / (numbins)
    bins = [minratio + i * binsize for i in range(numbins+ 1)]
    tickszie = (maxratio - minratio) / numticks
    ticks = [minratio + i * tickszie for i in range(numticks + 1)]

    lowborder = np.mean(odds_random) - 3 * np.std(odds_random)
    highborder = np.mean(odds_random) + 3 * np.std(odds_random)

    allheights = []

    x, b, ppred = P.hist(log_odds_of_induced, density=False, bins=bins, label="Predicted",
                         alpha=0.5, color=colorhist)


    for it in ppred:
        allheights.append(it.get_height())

    if len(odds_random) > 0:
        kernel = stats.gaussian_kde(odds_random)
        probs = kernel(log_odds_of_induced)
        # P.figure()
        stopsize = (maxratio - minratio) / 100
        stops = [minratio + stopsize * i for i in range(101)]
        hs = kernel(stops)
        lo = P.plot(stops, 1.5 * max(allheights) * hs / max(hs), '-', label="Random", color=colorrandom)


    if figsufix is not "":
        P.title(figsufix)
    if extratext is not None:
        P.legend(title=extratext)