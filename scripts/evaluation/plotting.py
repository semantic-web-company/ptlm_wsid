from typing import Set,List,Dict
from matplotlib import pylab as P
import numpy as np
from scipy import stats

def plot_against_randomized(log_odds_of_induced,
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