"""Adaptive k-AP attacks.

All attacks change at most k AP readings (set to a value in [1, 105], or
delete = 0). Success means the verifier accepts the impostor claim.

- beam_attack: model-agnostic beam search over single-AP edits.
- gradient_attack_mlp: gradient-guided L0 attack on a scikit-learn MLP
  (manual backpropagation), refined by exact forward evaluation.
- min_cost_attack_groups: exact attack on the group verifier over a value
  grid, exploiting that each AP affects one group (tightness of Theorem 1).
"""
import numpy as np

MAX_S = 105.0
VALUE_GRID = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])


def zone_profiles(X, y, Z):
    rate = np.zeros((Z, X.shape[1]))
    q50 = np.zeros((Z, X.shape[1]))
    q90 = np.zeros((Z, X.shape[1]))
    for z in range(Z):
        Xz = X[y == z]
        H = Xz > 0
        rate[z] = H.mean(0)
        for j in np.where(H.any(0))[0]:
            v = Xz[H[:, j], j]
            q50[z, j], q90[z, j] = np.median(v), np.percentile(v, 90)
    return rate, q50, q90


def candidate_edits(x, z, prof, n_add=60, n_del=25):
    rate, q50, q90 = prof
    add = np.argsort(-rate[z])[:n_add]
    add = add[rate[z, add] > 0]
    heard = np.where(x > 0)[0]
    dele = heard[np.argsort(rate[z, heard])][:n_del]
    edits = [(j, q50[z, j]) for j in add] + [(j, q90[z, j]) for j in add] + [(j, 0.0) for j in dele]
    return [(int(j), float(v)) for j, v in edits if x[j] != v]


def beam_attack(score_fn, accept_fn, x, z, prof, k_max, width=5):
    """Returns success[0..k_max] (bool). score_fn(Xb, z) -> higher = closer
    to acceptance; accept_fn(Xb, z) -> bool."""
    succ = np.zeros(k_max + 1, dtype=bool)
    succ[0] = accept_fn(x[None], z)[0]
    beams = [(x.copy(), frozenset())]
    for k in range(1, k_max + 1):
        if succ[k - 1]:
            succ[k:] = True
            break
        cand, keys = [], []
        for xb, used in beams:
            for j, v in candidate_edits(xb, z, prof):
                if j in used:
                    continue
                key = used | {j}
                xc = xb.copy(); xc[j] = v
                cand.append(xc); keys.append(key)
        if not cand:
            succ[k:] = succ[k - 1]
            break
        C = np.stack(cand)
        acc = accept_fn(C, z)
        if acc.any():
            succ[k:] = True
            break
        s = score_fn(C, z)
        top = np.argsort(-s)[:width]
        beams = [(C[i], keys[i]) for i in top]
    return succ


# ---------------------------------------------------------------------------
# Gradient attack on sklearn MLPClassifier (ReLU hidden, softmax output)
# ---------------------------------------------------------------------------
def mlp_logp_and_grad(mlp, A, z):
    """log p_z and its gradient w.r.t. the (scaled) input A (n, d)."""
    acts = [A]
    h = A
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0.0)
        acts.append(h)
    logits = h @ mlp.coefs_[-1] + mlp.intercepts_[-1]
    cls = list(mlp.classes_)
    zi = cls.index(z)
    if logits.shape[1] == 1:                             # binary: logistic output
        p1 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        pz = p1 if zi == 1 else 1.0 - p1
        g = ((1.0 - p1) if zi == 1 else -p1)[:, None]    # d log p_z / d logit
        p = np.c_[1.0 - p1, p1]
    else:
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        g = -p.copy(); g[:, zi] += 1.0                   # d log p_z / d logits
    g = g @ mlp.coefs_[-1].T
    for li in range(len(mlp.coefs_) - 2, -1, -1):
        g = g * (acts[li + 1] > 0)
        g = g @ mlp.coefs_[li].T
    return np.log(p[:, zi] + 1e-12), g


def gradient_attack_mlp(mlp, thr, x, z, k_max, scale=MAX_S, top=8):
    """Each step: rank APs by first-order gain of moving them to 80 (strong)
    or 0 (delete); evaluate the top candidates exactly; keep the best."""
    succ = np.zeros(k_max + 1, dtype=bool)
    xa = x.copy()
    used = set()
    prob = lambda X: mlp.predict_proba(X / scale)[:, list(mlp.classes_).index(z)]
    succ[0] = prob(xa[None])[0] >= thr
    for k in range(1, k_max + 1):
        if succ[k - 1]:
            succ[k:] = True
            break
        _, g = mlp_logp_and_grad(mlp, xa[None] / scale, z)
        g = g[0]
        gain_up = g * (80.0 - xa) / scale
        gain_dn = g * (0.0 - xa) / scale
        best = np.maximum(gain_up, gain_dn)
        best[list(used)] = -np.inf
        cands = np.argsort(-best)[:top]
        C = []
        for j in cands:
            for v in (80.0, 0.0, 50.0):
                xc = xa.copy(); xc[j] = v; C.append((j, xc))
        P = prob(np.stack([c[1] for c in C]))
        b = int(np.argmax(P))
        used.add(C[b][0]); xa = C[b][1]
        succ[k] = P[b] >= thr
    return succ


# ---------------------------------------------------------------------------
# Exact (over VALUE_GRID) minimum-cost attack on the one-sided group verifier
# ---------------------------------------------------------------------------
def group_flip_cost(gv, x, z, max_cost=2, grid=VALUE_GRID, rate=None, need=None):
    """cost[g] = min #APs (<= max_cost, values from grid) to make group g
    accept z; 0 if it already does; inf if not found.

    Candidate edits per group: delete a heard AP, or set an AP to a grid
    value. If `rate` (zone detection rates) is given, only APs heard in zone
    z or in x are edited (inserting an AP that zone z never hears moves the scan away
    from zone-z training points), so the search is exact over this restricted
    edit set and a lower bound on the optimal attack.
    Pairs are searched only if fewer than `need` groups flip with one edit.
    """
    A0 = gv.accept(x[None])[0, :, z]
    cost = np.full(gv.G, np.inf)
    cost[A0] = 0
    singles = {}
    for g in range(gv.G):
        if A0[g] or gv.data[g] is None:
            continue
        cols = gv.data[g][0]
        if rate is not None:
            cols = cols[(rate[z, cols] > 0) | (x[cols] > 0)]
        C1 = [x.copy() for _ in range(0)]
        for j in cols:
            for v in grid:
                if x[j] != v:
                    xc = x.copy(); xc[j] = v; C1.append(xc)
        if not C1:
            continue
        C1 = np.stack(C1)
        cnt = gv._counts(C1, g)[0][:, z]
        heard = C1[:, gv.data[g][0]].sum(1) > 0
        if ((cnt >= gv.m) & heard).any():
            cost[g] = 1
        else:
            singles[g] = (cols, C1[np.argsort(-cnt)[:6]])
    n_one = int(np.isfinite(cost).sum() - A0.sum())
    if max_cost >= 2 and (need is None or n_one < need):
        for g, (cols, seeds) in singles.items():
            C2 = []
            for s0 in seeds:
                changed = np.where(s0 != x)[0]
                for j in cols:
                    if j in changed:
                        continue
                    for v in grid:
                        if s0[j] != v:
                            xc = s0.copy(); xc[j] = v; C2.append(xc)
            if C2:
                if gv.accept_group(np.stack(C2), g)[:, z].any():
                    cost[g] = 2
    return cost


def min_cost_success(cost, v0, tau_tilde, u, k_max):
    """Attack succeeds with budget k if v0 + (#groups flippable within budget)
    + u >= tau_tilde, choosing cheapest groups first."""
    c = np.sort(cost[(cost > 0) & np.isfinite(cost)])
    succ = np.zeros(k_max + 1, dtype=bool)
    for k in range(k_max + 1):
        spent, gained = 0, 0
        for ci in c:
            if spent + ci > k:
                break
            spent += ci; gained += 1
        succ[k] = v0 + gained + u >= tau_tilde
    return succ


# ---------------------------------------------------------------------------
# Attack on a cascade (AP-Certified layer AND a second verifier)
# ---------------------------------------------------------------------------
def group_single_edits(gv, x, z, grid=VALUE_GRID, rate=None):
    """For each group that does not accept z, the single edits (j, v) that
    make it accept z."""
    A0 = gv.accept(x[None])[0, :, z]
    out = {}
    for g in range(gv.G):
        if A0[g] or gv.data[g] is None:
            continue
        cols = gv.data[g][0]
        if rate is not None:
            cols = cols[(rate[z, cols] > 0) | (x[cols] > 0)]
        cand = [(int(j), float(v)) for j in cols for v in grid if x[j] != v]
        if not cand:
            continue
        C = np.repeat(x[None], len(cand), 0)
        for r, (j, v) in enumerate(cand):
            C[r, j] = v
        ok = gv.accept_group(C, g)[:, z]
        if ok.any():
            out[g] = [cand[r] for r in np.where(ok)[0]]
    return out


def cascade_greedy_attack(edits, x, z, accept_fn, score2_fn, k_max):
    """Flip groups one AP at a time; among all single edits that flip a not
    yet flipped group, pick the one that most raises the second verifier's
    score. accept_fn(X, z) -> bool for the whole cascade."""
    succ = np.zeros(k_max + 1, dtype=bool)
    xa = x.copy()
    succ[0] = accept_fn(xa[None], z)[0]
    remaining = dict(edits)
    for k in range(1, k_max + 1):
        if succ[k - 1]:
            succ[k:] = True
            break
        cands = [(g, j, v) for g, lst in remaining.items() for j, v in lst if xa[j] != v]
        if not cands:
            succ[k:] = False
            break
        C = np.repeat(xa[None], len(cands), 0)
        for r, (g, j, v) in enumerate(cands):
            C[r, j] = v
        acc = accept_fn(C, z)
        if acc.any():
            succ[k:] = True
            break
        s = score2_fn(C, z)
        b = int(np.argmax(s))
        g, j, v = cands[b]
        xa = C[b]
        remaining.pop(g, None)
    return succ
