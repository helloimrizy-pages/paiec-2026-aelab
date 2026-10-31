"""H22 runtime rule (shared by simulation and submission): mixture over library configs of the subject's model + base model,
or an item-easiness model when the item is known but the subject's model is not.

mix_predict: hypotheses c (library configs of the subject's canonical model) each predict the target from their recorded
outcome on the target item; hypothesis 'base' predicts p_base. Labels revealed for this pair update the weights: config c
explains label y_j with its own recorded outcome on item j (if any; else the library item mean), base with the library item
mean (or 0.5). EPS bounds a recorded outcome away from 0/1.
easy_predict: logit p_i = logit(m_i) + u, u ~ N(0, TAU^2), Gauss-Hermite posterior from labeled items with known m_j.
"""
import math
import numpy as np
from numpy.polynomial.hermite import hermgauss

EPS = 0.03
TAU = 1.2
_gx, _gw = hermgauss(41); _gx, _gw = np.sqrt(2) * _gx, _gw / np.sqrt(np.pi)


def _clip(p, e=EPS):
    return min(1 - e, max(e, float(p)))


def priors(has_match: list, n_configs: int):
    """has_match[c]: config c matches the subject's harness/settings. Returns prior weights for configs and base."""
    if n_configs == 0:
        return [], 1.0
    if any(has_match):
        k = sum(has_match); rest = n_configs - k
        w = [0.75 / k if m else (0.10 / rest if rest else 0.0) for m in has_match]
        return w, 1.0 - sum(w)
    return [0.45 / n_configs] * n_configs, 0.55


def mix_predict(target_by_config, prior_w, prior_base, p_base, labeled):
    """target_by_config: list of recorded outcome (mean over trials) per config on the target item, or None.
    labeled: list of (y, outs_by_config list (None if missing), item_mean or None)."""
    logw = [math.log(w) if w > 0 else -1e9 for w in prior_w] + [math.log(prior_base) if prior_base > 0 else -1e9]
    for y, outs, m in labeled:
        q0 = _clip(m if m is not None else 0.5, 0.05)
        for c, o in enumerate(outs):
            q = _clip(o) if o is not None else q0
            logw[c] += math.log(q if y else 1 - q)
        logw[-1] += math.log(q0 if y else 1 - q0)
    mx = max(logw); w = [math.exp(v - mx) for v in logw]; s = sum(w); w = [v / s for v in w]
    p = w[-1] * p_base
    for c, o in enumerate(target_by_config):
        p += w[c] * (_clip(o) if o is not None else p_base)
    return float(p), w


def easy_predict(m_target, labeled_pairs, tau=TAU, u0=0.0):
    """m_target: library item mean (shrunk) for the target; labeled_pairs: list of (y, m_j)."""
    lt = math.log(m_target / (1 - m_target))
    u = u0 + tau * _gx; lw = np.log(_gw)
    for y, m in labeled_pairs:
        f = math.log(m / (1 - m)) + u
        lw = lw + (y * f - np.logaddexp(0, f))
    w = np.exp(lw - lw.max()); w /= w.sum()
    return float(w @ (1 / (1 + np.exp(-(lt + u)))))


def shrink(k, n, a=1.0, prior=0.5):
    return (k + a * prior) / (n + a)
