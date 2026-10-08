"""AP-Certified core: group verifier, certificates (k, delta), conformal calibration,
hierarchical verifier and partition analysis.

Conventions
- Scans are strength vectors s = RSSI + 105 (1..105); 0 means "not heard".
- A group g ACCEPTS zone z for scan x if at least one AP of g is heard and the
  group's kNN posterior P_g(z | x_g) >= theta. For the one-sided rule theta may
  be any value in (0, 1]; a group may then accept several zones, which the
  one-sided certificate allows (it is evaluated per claimed zone).
- Votes v_z(x) = number of groups that accept z.
"""
import math

import numpy as np
from scipy.stats import beta as beta_dist

MAX_S = 105.0


# =========================================================================
# Partitions
# =========================================================================
def partition_random(n_aps, G, seed=0):
    return np.random.default_rng(seed).integers(0, G, size=n_aps)


def partition_spread(X, G, seed=0, cols=None):
    """Greedy co-occurrence spreading (approximate rainbow colouring).

    Visit APs from most to least frequently heard; put each in the group
    with the smallest co-occurrence with APs already there. If `cols` is
    given, only those AP columns are partitioned and the rest get -1.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    cols = np.arange(n) if cols is None else np.asarray(cols)
    H = (X[:, cols] > 0).astype(np.float32)
    freq = H.sum(0)
    C = H.T @ H
    order = np.argsort(-freq + rng.uniform(0, 1e-3, size=freq.shape))
    groups = -np.ones(n, dtype=int)
    load = np.zeros((len(cols), G))
    size = np.zeros(G)
    for i in order:
        if freq[i] == 0:
            groups[cols[i]] = int(rng.integers(0, G))
            continue
        g = int(np.argmin(load[i] + 1e-6 * size))
        groups[cols[i]] = g
        size[g] += 1
        load[:, g] += C[:, i]
    return groups


def coverage(X, groups):
    """c(x) = number of distinct groups among the heard APs of each scan."""
    G = int(groups.max()) + 1
    M = np.zeros((X.shape[0], G), dtype=bool)
    for g in range(G):
        cols = np.where(groups == g)[0]
        if len(cols):
            M[:, g] = (X[:, cols] > 0).any(1)
    return M.sum(1)


def random_coverage_expectation(h, G):
    """E[c] for a uniform random partition when the scan hears h APs."""
    return G * (1.0 - (1.0 - 1.0 / G) ** np.asarray(h, dtype=float))


# =========================================================================
# Group verifier
# =========================================================================
class GroupVerifier:
    """One uniform kNN per AP group (columns restricted to the group)."""

    def __init__(self, groups, n_zones, K=10, theta=0.5, n_groups=None, power=1.0):
        self.power = power                 # "powed" representation f(s) = 105 (s/105)^power
        self.groups = np.asarray(groups)   # -1 = AP ignored by this verifier
        self.G = n_groups or int(self.groups.max()) + 1
        self.Z = n_zones
        self.K = K
        self.theta = theta
        self.m = int(math.ceil(theta * K - 1e-9))   # neighbours needed to accept
        self.data = [None] * self.G

    def fit(self, X, y):
        for g in range(self.G):
            cols = np.where(self.groups == g)[0]
            if len(cols) == 0:
                continue
            Xg = X[:, cols]
            rows = Xg.sum(1) > 0
            if rows.sum() < self.K:
                continue
            self.data[g] = (cols, self.f(Xg[rows]).astype(np.float32), y[rows].astype(int))
        return self

    def f(self, S):
        """Monotone per-AP feature map; 0 (not heard) stays 0."""
        return MAX_S * (np.asarray(S, dtype=np.float32) / MAX_S) ** self.power

    def _counts(self, Xq, g):
        """Neighbour class counts (n, Z) for scans whose group g is heard."""
        cols, T, yt = self.data[g]
        Q = self.f(Xq[:, cols])
        heard = Q.sum(1) > 0
        cnt = np.zeros((len(Q), self.Z), dtype=np.int16)
        idx = np.where(heard)[0]
        tn = (T ** 2).sum(1)
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            D = (Q[b] ** 2).sum(1)[:, None] + tn[None, :] - 2.0 * Q[b] @ T.T
            nn = np.argpartition(D, self.K - 1, axis=1)[:, :self.K]
            lab = yt[nn]
            for z in range(self.Z):
                cnt[b, z] = (lab == z).sum(1)
        return cnt, heard

    def accept(self, Xq):
        """A[n, g, z] = group g accepts zone z."""
        A = np.zeros((len(Xq), self.G, self.Z), dtype=bool)
        for g in range(self.G):
            if self.data[g] is None:
                continue
            cnt, heard = self._counts(Xq, g)
            A[:, g, :] = (cnt >= self.m) & heard[:, None]
        return A

    def accept_group(self, Xq, g):
        """(n, Z) acceptance of a single group."""
        cnt, heard = self._counts(Xq, g)
        return (cnt >= self.m) & heard[:, None]

    def votes(self, Xq):
        return self.accept(Xq).sum(1)

    def soft_score(self, Xq, z):
        """Sum over groups of P_g(z|x); a continuous score for attacks."""
        s = np.zeros(len(Xq))
        for g in range(self.G):
            if self.data[g] is None:
                continue
            cnt, heard = self._counts(Xq, g)
            s += cnt[np.arange(len(Xq)), z] / self.K
        return s

    # ---------------------------------------------------------------------
    # delta-certificate, pairwise version (tighter)
    # ---------------------------------------------------------------------
    def possible_accept_pairwise(self, Xq, delta):
        """Sound and tighter than `possible_accept`.

        For points b, s the gap d^2(x', b) - d^2(x', s) = -2 x'.(b - s) +
        |b|^2 - |s|^2 is linear in x', so its maximum over the box is exact.
        Take B = the K - m + 1 non-z training points nearest to x. If fewer
        than m z-points escape domination by every point of B (b always
        strictly closer than s), then at most m - 1 z-points can enter the
        K nearest neighbours for any admissible x', so the group rejects z.
        """
        n = len(Xq)
        P = np.zeros((n, self.G, self.Z), dtype=bool)
        nb = self.K - self.m + 1
        for g in range(self.G):
            if self.data[g] is None:
                continue
            cols, T, yt = self.data[g]
            Q = Xq[:, cols].astype(np.float32)
            heard = Q > 0
            lo = np.where(heard, self.f(np.maximum(Q - delta, 1.0)), 0.0)
            hi = np.where(heard, self.f(np.minimum(Q + delta, MAX_S)), 0.0)
            Fq = self.f(Q)
            tn = (T ** 2).sum(1)
            for i in np.where(heard.any(1))[0]:
                d0 = ((T - Fq[i]) ** 2).sum(1)
                for z in range(self.Z):
                    zm = yt == z
                    if zm.sum() < self.m:
                        continue
                    nz = np.where(~zm)[0]
                    if len(nz) < nb:
                        P[i, g, z] = True
                        continue
                    Bi = nz[np.argpartition(d0[nz], nb - 1)[:nb]]
                    S = T[zm]
                    # max over box of -2 x'.(b - s) + |b|^2 - |s|^2, for all (b, s)
                    D = T[Bi][:, None, :] - S[None, :, :]            # (nb, ns, d)
                    lin = np.maximum(-2 * lo[i] * D, -2 * hi[i] * D).sum(2)
                    gap = lin + tn[Bi][:, None] - tn[zm][None, :]
                    dominated = (gap < 0).all(0)                     # by every b in B
                    P[i, g, z] = (~dominated).sum() >= self.m
        return P

    # ---------------------------------------------------------------------
    # delta-certificate for the bounded-offset part of the threat model
    # ---------------------------------------------------------------------
    def possible_accept(self, Xq, delta):
        """P[n, g, z] = True unless we can PROVE that no x' with
        |x'_j - x_j| <= delta on heard APs (unheard APs stay unheard;
        heard APs stay within [1, 105]) makes group g accept z.

        Sound test: with distance bounds L_t <= d(x', t) <= U_t for every
        training point t, if at least K - m + 1 non-z points satisfy
        U_t < (m-th smallest L among z points), then fewer than m z points
        can be among the K nearest neighbours for every admissible x'.
        """
        n = len(Xq)
        P = np.zeros((n, self.G, self.Z), dtype=bool)
        for g in range(self.G):
            if self.data[g] is None:
                continue
            cols, T, yt = self.data[g]
            Q = Xq[:, cols].astype(np.float32)
            heard = Q > 0
            # interval in strength space, mapped through the monotone feature map
            lo = np.where(heard, self.f(np.maximum(Q - delta, 1.0)), 0.0)
            hi = np.where(heard, self.f(np.minimum(Q + delta, MAX_S)), 0.0)
            zmask = [yt == z for z in range(self.Z)]
            for i in np.where(heard.any(1))[0]:
                a, b = lo[i][None, :], hi[i][None, :]
                below, above = T < a, T > b
                dmin = np.where(below, a - T, np.where(above, T - b, 0.0))
                dmax = np.maximum(np.abs(T - a), np.abs(T - b))
                L = (dmin ** 2).sum(1)
                U = (dmax ** 2).sum(1)
                for z in range(self.Z):
                    Lz = L[zmask[z]]
                    if len(Lz) < self.m:
                        continue
                    thr = np.partition(Lz, self.m - 1)[self.m - 1]
                    blockers = (U[~zmask[z]] < thr).sum()
                    P[i, g, z] = blockers < self.K - self.m + 1
        return P


# =========================================================================
# Certificates
# =========================================================================
def one_sided_certified_reject(v, tau, k):
    """Theorem 1 (deterministic threshold): v + k < tau."""
    return v + k < tau


def certified_far_randomized(v_imp, tau_tilde, k):
    """Randomized threshold: accept iff v + u >= tau_tilde, u ~ U(0,1) drawn by
    the server. For a (k, delta) adversary v' <= v_delta + k, so the attack can
    succeed only if v_delta + k + u >= tau_tilde. Returns the exact certified
    FAR (expectation over u) averaged over impostor claims."""
    return np.clip(v_imp + k + 1 - tau_tilde, 0.0, 1.0).mean()


# =========================================================================
# Conformal calibration (randomized, for integer vote scores)
# =========================================================================
def conformal_tau(gen_scores, alpha, rng, jitter=1.0):
    """Split-conformal threshold with randomized tie-breaking.

    Scores s~ = s + u, u ~ U(0,1) i.i.d. Let j = floor(alpha (n+1)); the
    threshold is the j-th smallest jittered calibration score (or -inf if
    j = 0). For an exchangeable new genuine claim, P(s~_new < tau) <= alpha.
    """
    n = len(gen_scores)
    j = int(math.floor(alpha * (n + 1)))
    if j < 1:
        return -np.inf
    st = np.sort(gen_scores + jitter * rng.uniform(0, 1, n))
    return st[j - 1]


def conformal_tau_pac(gen_scores, alpha, beta, rng, jitter=1.0):
    """Training-conditional version: largest j with
    P(Beta(j, n-j+1) <= alpha) >= 1 - beta, so FRR <= alpha with prob >= 1-beta
    over the calibration draw."""
    n = len(gen_scores)
    js = np.arange(1, n + 1)
    ok = js[beta_dist.cdf(alpha, js, n - js + 1) >= 1 - beta]
    if len(ok) == 0:
        return -np.inf
    st = np.sort(gen_scores + jitter * rng.uniform(0, 1, n))
    return st[ok.max() - 1]


def frr_far(gen, imp, tau, rng, jitter=1.0):
    """jitter = 1 for integer vote scores (randomized threshold); a tiny
    value for continuous scores such as probabilities."""
    ug, ui = jitter * rng.uniform(0, 1, len(gen)), jitter * rng.uniform(0, 1, len(imp))
    return np.mean(gen + ug < tau), np.mean(imp + ui >= tau)


def eer(gen, imp):
    thr = np.unique(np.concatenate([gen, imp]))
    best, val = 2.0, None
    for t in thr:
        frr, far = np.mean(gen < t), np.mean(imp >= t)
        if abs(frr - far) < best:
            best, val = abs(frr - far), (frr + far) / 2
    return val


# =========================================================================
# Hierarchical verifier (building, then floor)
# =========================================================================
class HierarchicalVerifier:
    """Score for claim (b, f): s = min(v_B(x, b), v_F^b(x, f)).

    Stage B uses groups over all APs. Stage F for building b uses groups over
    the APs heard in building b only. Each AP lies in at most one stage-B
    group and at most one stage-F group of a given building, so a k-AP
    adversary raises each vote count, and hence s, by at most k.
    """

    def __init__(self, GB, GF, K=10, thetaB=0.5, thetaF=0.5, seed=0, power=1.0, KF=None):
        self.GB, self.GF, self.K, self.power, self.KF = GB, GF, K, power, (KF or K)
        self.thetaB, self.thetaF, self.seed = thetaB, thetaF, seed

    def fit(self, X, building, floor):
        self.buildings = np.unique(building)
        self.nF = int(floor.max()) + 1
        gB = partition_spread(X, self.GB, self.seed)
        self.stageB = GroupVerifier(gB, len(self.buildings), self.K, self.thetaB, power=self.power).fit(X, building)
        self.stageF = {}
        for b in self.buildings:
            r = building == b
            cols = np.where((X[r] > 0).any(0))[0]
            gF = partition_spread(X[r], self.GF, self.seed, cols=cols)  # -1 outside building b
            v = GroupVerifier(gF, self.nF, self.KF, self.thetaF, n_groups=self.GF, power=self.power).fit(X[r], floor[r])
            self.stageF[b] = v
        return self

    def claim_scores(self, Xq, delta=0.0):
        """S[n, b, f] = min(vB(b), vF_b(f)); with delta > 0 the counts use
        possible acceptances (upper bounds on votes under bounded offsets)."""
        if delta > 0:
            AB = self.stageB.possible_accept(Xq, delta)
        else:
            AB = self.stageB.accept(Xq)
        vB = AB.sum(1)
        S = np.zeros((len(Xq), len(self.buildings), self.nF))
        for b in self.buildings:
            st = self.stageF[b]
            AF = st.possible_accept(Xq, delta) if delta > 0 else st.accept(Xq)
            vF = AF.sum(1)
            S[:, b, :] = np.minimum(vB[:, b][:, None], vF)
        return S


# =========================================================================
# Multi-scan logins
# =========================================================================
def merge_scans(S):
    """Merge m scans of one login into one fingerprint (n_aps,).
    An AP is heard if heard in any scan; its strength is the mean over the
    scans that heard it. Editing AP j in any scan changes only entry j of the
    merged fingerprint, and per-scan offsets of at most delta move the mean
    by at most delta, so Theorems 1-2 apply unchanged to the merged input."""
    S = np.asarray(S, dtype=np.float32)
    H = S > 0
    cnt = H.sum(0)
    out = np.zeros(S.shape[1], dtype=np.float32)
    nz = cnt > 0
    out[nz] = S[:, nz].sum(0) / cnt[nz]
    return out


# =========================================================================
# Hybrid rule: one-sided votes and plurality margin in one 1-Lipschitz score
# =========================================================================
def margin_matrix(V):
    """M[n, z] = V[n, z] - max_{z' != z} V[n, z']."""
    V = np.asarray(V, dtype=float)
    M = np.empty_like(V)
    for z in range(V.shape[1]):
        M[:, z] = V[:, z] - np.delete(V, z, axis=1).max(1)
    return M


def hybrid_scores(V, c):
    """s = min(v_z, (margin_z + c) / 2).  With exclusive group votes
    (theta > 1/2) a changed group moves v_z by <= 1 and the margin by <= 2,
    so s moves by <= 1 per manipulated AP. c = +inf gives the one-sided
    rule, c = -inf the symmetric margin rule (up to a constant)."""
    V = np.asarray(V, dtype=float)
    if np.isposinf(c):
        return V
    M = margin_matrix(V)
    if np.isneginf(c):
        return M / 2.0
    return np.minimum(V, (M + c) / 2.0)


def weighted_votes(A, w):
    """A[n, g, z] acceptances, w[g] positive integer weights."""
    return np.einsum("ngz,g->nz", A.astype(float), w)


def topk_weight(w, k):
    """Largest total weight an adversary can flip with k APs."""
    ws = np.sort(np.asarray(w))[::-1]
    return float(ws[:k].sum())


def certified_far_weighted(v_imp, tau, w, k):
    return np.clip(v_imp + topk_weight(w, k) + 1 - tau, 0.0, 1.0).mean()


# =========================================================================
# Online calibration under drift (quantile tracking with a security floor)
# =========================================================================
class OnlineThreshold:
    """tau_{t+1} = tau_t - eta (err_t - alpha) after each genuine attempt
    whose outcome is later confirmed (err_t = 1 if it was rejected).

    Guarantee (Theorem 4): for any sequence of jittered genuine scores in
    [a, a + R], |(1/T) sum err_t - alpha| <= (R + 2 eta) / (eta T) when no
    floor is used (run_v3.part_C implements the same update inline). With a floor tau_min, rounds where the floor binds are reported as
    drift alarms: the FRR bound may then be violated, but the certified
    radius implied by tau_min is kept.
    """

    def __init__(self, tau0, alpha=0.05, eta=0.05, tau_min=None):
        self.tau, self.alpha, self.eta, self.tau_min = float(tau0), alpha, eta, tau_min
        self.alarm = False

    def update(self, err):
        t = self.tau - self.eta * (float(err) - self.alpha)
        self.alarm = self.tau_min is not None and t < self.tau_min
        self.tau = max(t, self.tau_min) if self.tau_min is not None else t
        return self.tau


# =========================================================================
# Population-level certified FAR (Clopper-Pearson)
# =========================================================================
def cp_upper(x, n, beta):
    """One-sided Clopper-Pearson upper confidence bound for a binomial rate."""
    from scipy.stats import beta as _beta
    if n == 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(_beta.ppf(1 - beta, x + 1, n - x))


def certified_far_upper(v_imp, tau_tilde, ks, rng, beta=0.05, groups=None):
    """Upper confidence bounds on the population certified FAR for each k in ks.

    For impostor claim i with a fresh secret draw u_i, the indicator
    B_i(k) = 1[v_i + k + u_i >= tau] dominates the success indicator of every
    (k, 0) adversary (Theorem 1), and E[B_i(k)] equals the population certified
    FAR. With independent claims, a Clopper-Pearson bound on sum_i B_i(k) bounds
    it with confidence 1 - beta; beta is split over len(ks) radii (Bonferroni),
    so the bounds hold simultaneously. If groups is given (e.g. survey
    positions), one claim per group is drawn at random so that the retained
    claims are independent across groups.
    """
    v = np.asarray(v_imp, float)
    if groups is not None:
        groups = np.asarray(groups)
        keep = []
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            keep.append(rng.choice(idx))
        v = v[np.array(keep)]
    u = rng.uniform(0, 1, len(v))
    b = beta / len(ks)
    return [cp_upper(int((v + k + u >= tau_tilde).sum()), len(v), b) for k in ks]
