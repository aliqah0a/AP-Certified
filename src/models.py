"""Verifiers, AP partitions, certificates and attacks for AP-Certified."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------------------------------
# AP partitions
# --------------------------------------------------------------------------
def partition_random(n_aps, G, seed=0):
    """Random hash partition: AP j -> group h(j)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, G, size=n_aps)


def partition_spread(X, G, seed=0):
    """Co-occurrence-spreading partition.

    APs that are often heard together are placed in different groups, so a
    single scan touches as many groups as possible. Greedy: visit APs from
    most to least frequently heard; put each AP in the group with the lowest
    total co-occurrence with APs already assigned there (ties: smallest group).
    """
    rng = np.random.default_rng(seed)
    H = (X > 0).astype(np.float32)
    freq = H.sum(0)
    C = H.T @ H  # co-occurrence counts
    order = np.argsort(-freq + rng.uniform(0, 1e-3, size=freq.shape))
    groups = -np.ones(X.shape[1], dtype=int)
    load = np.zeros((X.shape[1], G))  # load[j, g] = co-occ of AP j with group g
    size = np.zeros(G)
    for j in order:
        if freq[j] == 0:
            groups[j] = rng.integers(0, G)
            continue
        cost = load[j] + 1e-6 * size
        g = int(np.argmin(cost))
        groups[j] = g
        size[g] += 1
        load[:, g] += C[:, j]
    return groups


# --------------------------------------------------------------------------
# AP-partitioned vote verifier
# --------------------------------------------------------------------------
class GroupVoteVerifier:
    """One light classifier per AP group; each group votes for one zone.

    A group with no heard AP in the scan abstains (votes for nothing).
    With theta=None a heard group votes for its predicted zone. With
    theta in (0.5, 1] it votes for zone z only if P_g(z | x_g) >= theta,
    otherwise it abstains. Either way a group casts at most one vote, and
    the vote depends only on the readings of that group's APs.
    """

    def __init__(self, groups, n_zones, n_neighbors=5, theta=None):
        self.groups = np.asarray(groups)
        self.G = int(self.groups.max()) + 1
        self.Z = n_zones
        self.k = n_neighbors
        self.theta = theta
        assert theta is None or theta > 0.5, "theta must exceed 0.5 (one vote per group)"
        self.models = [None] * self.G

    def fit(self, X, y):
        for g in range(self.G):
            cols = np.where(self.groups == g)[0]
            if len(cols) == 0:
                continue
            Xg = X[:, cols]
            rows = Xg.sum(1) > 0
            if rows.sum() < self.k or len(np.unique(y[rows])) < 1:
                continue
            m = KNeighborsClassifier(n_neighbors=self.k,
                                     weights="distance" if self.theta is None else "uniform")
            m.fit(Xg[rows], y[rows])
            self.models[g] = (cols, m)
        return self

    def group_predictions(self, X):
        """P[n, g] = zone voted by group g, or -1 if the group abstains."""
        P = -np.ones((X.shape[0], self.G), dtype=int)
        for g, entry in enumerate(self.models):
            if entry is None:
                continue
            cols, m = entry
            Xg = X[:, cols]
            rows = np.where(Xg.sum(1) > 0)[0]
            if not len(rows):
                continue
            if self.theta is None:
                P[rows, g] = m.predict(Xg[rows])
            else:
                pr = m.predict_proba(Xg[rows])
                best = pr.argmax(1)
                ok = pr[np.arange(len(rows)), best] >= self.theta
                P[rows[ok], g] = m.classes_[best[ok]]
        return P

    def votes(self, X):
        P = self.group_predictions(X)
        V = np.zeros((X.shape[0], self.Z), dtype=int)
        for g in range(self.G):
            ok = P[:, g] >= 0
            np.add.at(V, (np.where(ok)[0], P[ok, g]), 1)
        return V


# --------------------------------------------------------------------------
# Decision rules and certificates
# --------------------------------------------------------------------------
def one_sided_accept(v_claim, tau):
    return v_claim >= tau


def one_sided_certified_reject(v_claim, tau, k):
    """Theorem 1: each manipulated AP lies in exactly one group, so k APs
    change at most k group votes. If v + k < tau, no (k, 0) attacker can
    reach the threshold."""
    return v_claim + k < tau


def symmetric_margin(V, z):
    """Margin of the claimed zone over the best other zone."""
    Vo = V.copy()
    Vo[np.arange(len(z)), z] = -1
    top_other = Vo.max(1)
    return V[np.arange(len(z)), z] - top_other, Vo


def symmetric_certified_reject(V, z, m, k):
    """Plurality-margin rule (FPA style): accept iff V[z] - max_other >= m.

    Worst case for k flipped groups: z gains k votes and the other zones
    lose up to k votes in total, spread to minimise the strongest other
    zone (greedy water-filling: remove one vote from the current maximum,
    k times). An earlier version removed all k votes from the top zone only;
    the brute-force test showed that bound is unsound when two other zones
    tie, so it was replaced.
    """
    n = len(z)
    vz = V[np.arange(n), z]
    Vo = V.astype(int).copy()
    Vo[np.arange(n), z] = 0
    for _ in range(k):
        j = Vo.argmax(1)
        Vo[np.arange(n), j] = np.maximum(Vo[np.arange(n), j] - 1, 0)
    worst_other = Vo.max(1)
    return (vz + k) - worst_other < m


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def eer(gen, imp):
    """Equal error rate from genuine and impostor scores (higher = accept)."""
    thr = np.unique(np.concatenate([gen, imp]))
    best = (1.0, None)
    for t in thr:
        frr = np.mean(gen < t)
        far = np.mean(imp >= t)
        d = abs(frr - far)
        if d < best[0]:
            best = (d, (frr + far) / 2)
    return best[1]


def threshold_at_frr(gen, target=0.05):
    """Largest threshold whose FRR does not exceed the target."""
    thr = np.sort(np.unique(gen))[::-1]
    for t in thr:
        if np.mean(gen < t) <= target:
            return t
    return thr[-1]


# --------------------------------------------------------------------------
# Greedy k-AP attack on score-based verifiers
# --------------------------------------------------------------------------
def zone_profiles(X, y, Z):
    """Per zone: detection rate and median strength of each AP."""
    rate = np.zeros((Z, X.shape[1]))
    med = np.zeros((Z, X.shape[1]))
    for z in range(Z):
        Xz = X[y == z]
        H = Xz > 0
        rate[z] = H.mean(0)
        for j in np.where(H.any(0))[0]:
            med[z, j] = np.median(Xz[H[:, j], j])
    return rate, med


def greedy_attack(score_fn, x, z, rate, med, k_max, n_add=40, n_del=20):
    """White-box greedy attack: at each step, set or delete the single AP
    reading that most increases score_fn(x', z). Returns scores after
    0..k_max manipulated APs."""
    x = x.copy()
    add_c = np.argsort(-rate[z])[:n_add]
    scores = [float(score_fn(x[None], np.array([z]))[0])]
    used = set()
    for _ in range(k_max):
        heard = np.where(x > 0)[0]
        del_c = heard[np.argsort(rate[z, heard])][:n_del]
        cands = []
        for j in add_c:
            if j in used:
                continue
            xc = x.copy(); xc[j] = med[z, j]; cands.append((j, xc))
        for j in del_c:
            if j in used:
                continue
            xc = x.copy(); xc[j] = 0.0; cands.append((j, xc))
        if not cands:
            break
        Xc = np.stack([c[1] for c in cands])
        s = score_fn(Xc, np.full(len(cands), z))
        b = int(np.argmax(s))
        used.add(cands[b][0])
        x = cands[b][1]
        scores.append(float(s[b]))
    return np.array(scores)
