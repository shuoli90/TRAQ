import numpy as np
from scipy.stats.distributions import chi2
from scipy.stats import norm
from scipy.stats import combine_pvalues
import torch as tc
from torch import Tensor as T
from scipy.stats import hmean


def convert_rank(confidences, labels):
    indices = np.argsort(confidences)
    classes = np.stack(
        np.tile(np.arange(confidences.shape[1]).reshape(-1, 1),
                labels.shape[0])).T
    classes = classes[np.arange(confidences.shape[1]), indices]
    ranks = np.where(classes == labels.reshape(-1, 1).reshape(-1, 1))[-1]
    return ranks


def convert_quantile(confidences, labels):
    true_confidences = confidences[np.arange(confidences.shape[0]), labels]
    confidences = confidences.flatten()
    num_confidences = confidences.shape[0]
    confidences = np.array([confidences] * true_confidences.shape[0])
    true_confidences = np.array([true_confidences]*num_confidences).T
    quantiles = np.sum(true_confidences > confidences,
                       axis=1) / num_confidences
    return quantiles


def identify_rank(d_x, label):
    indices = np.argsort(d_x)
    rank = np.argwhere(indices == label)[0]
    return rank


def log_factorial(n):
    log_f = tc.arange(n, 0, -1).float().log().sum()
    return log_f


def log_n_choose_k(n, k):
    if k == 0:
        # return tc.tensor(1)
        return 0
    else:
        # res = log_factorial(n) - log_factorial(k) - log_factorial(n-k)
        # res = tc.arange(n, n-k, -1).float().log().sum() - log_factorial(k)
        res = np.sum(np.log(np.arange(n, n-k, -1.0))) - log_factorial(k)
        return res


def half_line_bound_upto_k(n, k, eps):
    ubs = []
    eps = tc.tensor(eps)
    for i in tc.arange(0, k+1):
        bc_log = log_n_choose_k(n, i)
        log_ub = bc_log + eps.log()*i + (1.0-eps).log()*(n-i)
        ubs.append(log_ub.exp().unsqueeze(0))
    ubs = tc.cat(ubs)
    ub = ubs.sum()
    return ub


def find_maximum_train_error_allow(eps, delta, n):
    k_min = 0
    k_max = n
    bnd_min = half_line_bound_upto_k(n, k_min, eps)
    if bnd_min > delta:
        return None
    assert (bnd_min <= delta)
    k = n
    while True:
        # choose new k
        k_prev = k
        k = (T(k_min + k_max).float()/2.0).round().long().item()

        # terinate condition
        if k == k_prev:
            break

        # check whether the current k satisfies the condition
        bnd = half_line_bound_upto_k(n, k, eps)
        if bnd <= delta:
            k_min = k
        else:
            k_max = k

    # confirm that the solution satisfies the condition
    k_best = k_min
    assert (half_line_bound_upto_k(n, k_best, eps) <= delta)
    error_allow = float(k_best) / float(n)
    return error_allow


def cal_pvalues(cals, scores):
    cal = np.tile(cals, (scores.shape[0], 1))
    score = np.tile(scores.reshape(-1, 1), (1, cal.shape[1]))
    p_values = np.mean(cal <= score, axis=1)
    return p_values


def harmonic_mean_p_value(p_values, weights=None):
    harmonic_p_values = hmean(p_values, weights=weights)
    return harmonic_p_values


def fisher_threshold(cals, scores, epsilon):
    p_values = 0
    for score, cal in zip(scores, cals):
        # cal = np.tile(cal.numpy(), (score.shape[0], 1))
        cal = np.tile(cal, (score.shape[0], 1))
        score = np.tile(score.reshape(-1, 1), (1, cal.shape[1]))
        p_values += np.log(np.mean(cal <= score, axis=1))
    tmp = -2 * p_values
    combined_p_values = 1 - chi2.cdf(tmp, df=2*len(cals))
    return np.quantile(combined_p_values, epsilon)


def fisher_test(cals, scores, threshold):
    p_values = 0
    for score, cal in zip(scores, cals):
        # cal = np.tile(cal.numpy(), (score.shape[0], 1))
        cal = np.tile(cal, (score.shape[0], 1))
        score = np.tile(score.reshape(-1, 1), (1, cal.shape[1]))
        p_values += np.log(np.mean(cal <= score, axis=1))
    tmp = -2 * p_values
    combined_p_values = 1 - chi2.cdf(tmp, df=2*len(cals))
    # breakpoint()
    return combined_p_values > threshold


def ECF(pvalues):
    # Extended Chi-Square Fucntion
    # The output is the significance level for the set of p-values
    k = np.prod(pvalues)
    m = len(pvalues)
    p_combined = 0
    for i in range(m):
        p_combined += k * (-np.log(k)) ** i / np.math.factorial(i)
    # p_combined is the significance level
    sign_level = p_combined
    return sign_level


def Fisher(pvalues):
    # Fisher's method
    m = len(pvalues)
    p_combined = 0
    for pvalue in pvalues:
        p_combined += -2 * np.log(pvalue)
    # Compute the significance level
    sign_level = 1 - chi2.cdf(p_combined, df=2*m)
    return sign_level


def SNF(pvalues):
    # Standard Normal Fucntion (Z-score)
    qs = 0
    m = len(pvalues)
    sn = norm(0, 1)
    for pvalue in pvalues:
        qs += 1 - sn.cdf(pvalue)
    sign_level = 1 - sn.cdf(qs / np.sqrt(m))
    return sign_level


def stouffer(pvalues):
    sign_level = combine_pvalues(pvalues.flatten(), method='stouffer')[1]
    return sign_level


def NCA(cals_cms, cms):
    # Non-conformity Aggregation
    C = np.sum(cms)
    cals_cms = np.sum(cals_cms, axis=0)
    sign_level = np.sum(cals_cms <= C) / cals_cms.shape[0]
    return sign_level


def SCP(cal_cms, cms):
    # Synergy Conformal Predictor
    cal_cms = np.sum(cal_cms, 0)
    cms = np.sum(cms)
    sign_level = np.sum(cal_cms < cms) / cal_cms.shape[0]
    return sign_level


if __name__ == "__main__":
    # Generate NCMS and pvalues
    cal_cms = np.abs(np.random.normal(loc=0.0, size=[5, 1000]))
    cms = np.abs(np.random.normal(loc=0.0, size=[5, 1]))
    pvalues = cal_pvalues(cal_cms, cms)

    print('ECF', ECF(pvalues))
    print('Fisher', Fisher(pvalues))
    print('SNF', SNF(pvalues))
    print('Stouffer', combine_pvalues(pvalues.flatten(), method='stouffer')[1])
    print('NCA', NCA(cal_cms, cms))
    print('SCP', SCP(cal_cms, cms))