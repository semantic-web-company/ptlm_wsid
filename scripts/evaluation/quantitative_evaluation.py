from scipy import stats
import numpy as np

def oddsratios_probs_vs_random(log_odds_of_induced,
                               randomized_logodds):
    odds_random = [item for sublist in randomized_logodds for item in sublist]
    kernel = stats.gaussian_kde(odds_random)
    probs = kernel(log_odds_of_induced)

    return np.mean(probs)
