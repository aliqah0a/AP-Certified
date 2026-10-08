"""Shared experiment helpers: baselines, claim matrices, evaluation."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from core import (GroupVerifier, certified_far_randomized, conformal_tau, coverage,
                  eer, frr_far, partition_random, partition_spread,
                  random_coverage_expectation)

K_RANGE = list(range(0, 11))
DELTAS = [0.0, 2.0, 4.0, 6.0]
ALPHA = 0.05
FAR_BUDGET = 0.01
SCALE = 105.0


def make_baselines(seed):
    return {
        "kNN": KNeighborsClassifier(n_neighbors=10),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   verbose=-1, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                             early_stopping=True, random_state=seed),
    }


def fit_baselines(X, y, seed=0):
    out = {}
    for name, m in make_baselines(seed).items():
        m.fit(X / SCALE, y)
        out[name] = m
    return out


def baseline_scores(m, X, Z):
    P = np.zeros((len(X), Z))
    P[:, m.classes_] = m.predict_proba(X / SCALE)
    return P


def split_claims(S, y):
    """Genuine and impostor scores from a (n, Z) claim-score matrix."""
    G = np.zeros_like(S, dtype=bool)
    G[np.arange(len(y)), y] = True
    return S[G], S[~G]


def kstar_from_curve(curve):
    """Largest k such that certified FAR <= budget for every radius up to k."""
    k_star = -1
    for c in curve:
        if c > FAR_BUDGET:
            break
        k_star += 1
    return k_star


def evaluate_split(Sc, yc, Se, ye, rng, mondrian_groups=None, extra_imp=None, jitter=1.0, tau=None):
    """Conformal threshold from calibration claims; metrics on evaluation claims.
    Returns dict with FRR, FAR, EER, tau, certified curve over K_RANGE.
    extra_imp: dict name -> impostor score array for additional certified curves
    (e.g. delta > 0 vote upper bounds)."""
    gc, _ = split_claims(Sc, yc)
    ge, ie = split_claims(Se, ye)
    if tau is None:
        tau = conformal_tau(gc, ALPHA, rng, jitter)
    frr, far = frr_far(ge, ie, tau, rng, jitter)
    res = dict(tau=tau, FRR=frr, FAR=far, EER=eer(ge, ie))
    if jitter == 1.0:
        res["cert"] = [certified_far_randomized(ie, tau, k) for k in K_RANGE]
    if extra_imp:
        for name, imp in extra_imp.items():
            res["cert_" + name] = [certified_far_randomized(imp, tau, k) for k in K_RANGE]
    if mondrian_groups is not None:
        mc, me = mondrian_groups
        frrs = {}
        for grp in np.unique(mc):
            gcg, _ = split_claims(Sc[mc == grp], yc[mc == grp])
            geg, _ = split_claims(Se[me == grp], ye[me == grp])
            t = conformal_tau(gcg, ALPHA, rng, jitter)
            frrs[int(grp)] = float(np.mean(geg + jitter * rng.uniform(0, 1, len(geg)) < t))
        res["mondrian_FRR"] = frrs
    return res


def q_alpha(c, alpha=ALPHA):
    """Q_alpha(c) = min{q : P[c <= q] > alpha} (Proposition 1)."""
    c = np.sort(np.asarray(c))
    vals = np.unique(c)
    for q in vals:
        if np.mean(c <= q) > alpha:
            return float(q)
    return float(vals[-1])


def partition_table(X_gen, X_fit, Gs, seed=0):
    """Coverage quantiles: upper bound min(G,|H|), random (empirical and
    expected), spread. Returned as list of dicts."""
    h = (X_gen > 0).sum(1)
    rows = []
    for G in Gs:
        pr = partition_random(X_fit.shape[1], G, seed)
        ps = partition_spread(X_fit, G, seed)
        cr, cs = coverage(X_gen, pr), coverage(X_gen, ps)
        rows.append(dict(G=G,
                         bound_q05=q_alpha(np.minimum(G, h)),
                         random_expected_q05=q_alpha(random_coverage_expectation(h, G)),
                         random_q05=q_alpha(cr),
                         spread_q05=q_alpha(cs),
                         bound_mean=float(np.minimum(G, h).mean()),
                         random_mean=float(cr.mean()), spread_mean=float(cs.mean())))
    return rows


def group_claim_votes(gv, X, delta=0.0):
    A = gv.possible_accept(X, delta) if delta > 0 else gv.accept(X)
    return A.sum(1).astype(float)
