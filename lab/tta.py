"""Test-time adaptation: Gaussian-process classification per subject-benchmark pair.

Latent logit f_i = m_i + g_i, where m_i is a fixed prior mean (pair prior logit
plus text-predicted item easiness) and g ~ GP(0, K) with an additive kernel

    K(i, j) = s0^2                                  (shared pair offset)
            + sg^2 * |tokens_i & tokens_j|          (shared item-feature groups)
            + se^2 * cos(emb_i, emb_j)^+ ** power   (text similarity)

The fit uses only the pair's revealed labels; a prediction uses only the target
item and those labels, as the competition requires. Laplace approximation and
probit-approximated predictive (Rasmussen & Williams, Algorithms 3.1 and 3.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import expit


@dataclass
class KernelParams:
    s0: float = 1.5
    sg: float = 1.0
    se: float = 1.0
    power: float = 4.0
    sz: float = 0.0  # prior s.d. of a per-pair slope on the scalar item feature z


def feature_tokens(item_features: str) -> frozenset[str]:
    return frozenset(t.strip() for t in (item_features or "").split(";") if "=" in t and t.strip())


def kernel(A_tok, A_emb, B_tok, B_emb, kp: KernelParams, A_z=None, B_z=None) -> np.ndarray:
    K = np.full((len(A_tok), len(B_tok)), kp.s0 ** 2)
    if kp.sz > 0 and A_z is not None:
        K += kp.sz ** 2 * np.outer(A_z, B_z)
    if kp.sg > 0:
        K += kp.sg ** 2 * np.array([[len(a & b) for b in B_tok] for a in A_tok], dtype=float)
    if kp.se > 0 and A_emb is not None:
        c = np.clip(A_emb @ B_emb.T, 0.0, 1.0)
        K += kp.se ** 2 * c ** kp.power
    return K


def kernel_diag(A_tok, A_emb, kp: KernelParams, A_z=None) -> np.ndarray:
    d = np.full(len(A_tok), kp.s0 ** 2)
    if kp.sz > 0 and A_z is not None:
        d += kp.sz ** 2 * np.asarray(A_z) ** 2
    if kp.sg > 0:
        d += kp.sg ** 2 * np.array([len(a) for a in A_tok], dtype=float)
    if kp.se > 0 and A_emb is not None:
        d += kp.se ** 2
    return d


class GPPair:
    """Laplace-approximated GP classifier for one pair."""

    def __init__(self, tok, emb, y, m, kp: KernelParams, iters: int = 30, z=None):
        self.tok, self.emb, self.kp, self.z = tok, emb, kp, z
        n = len(y)
        self.n = n
        if n == 0:
            return
        K = kernel(tok, emb, tok, emb, kp, z, z) + 1e-6 * np.eye(n)
        f = m.copy()
        for _ in range(iters):
            pi = expit(f)
            W = pi * (1 - pi)
            sW = np.sqrt(W)
            L = cholesky(np.eye(n) + sW[:, None] * K * sW[None, :], lower=True)
            b = W * (f - m) + (y - pi)
            a = b - sW * cho_solve((L, True), sW * (K @ b))
            f_new = K @ a + m
            if np.max(np.abs(f_new - f)) < 1e-6:
                f = f_new
                break
            f = f_new
        pi = expit(f)
        W = pi * (1 - pi)
        self.sW = np.sqrt(W)
        self.L = cholesky(np.eye(n) + self.sW[:, None] * K * self.sW[None, :], lower=True)
        self.resid = y - pi

    def predict(self, tok, emb, m, z=None) -> np.ndarray:
        kss = kernel_diag(tok, emb, self.kp, z)
        if self.n == 0:
            return expit(m / np.sqrt(1 + np.pi * kss / 8))
        ks = kernel(self.tok, self.emb, tok, emb, self.kp, self.z, z)  # n x t
        mean = m + ks.T @ self.resid
        v = solve_triangular(self.L, self.sW[:, None] * ks, lower=True)
        var = np.maximum(kss - np.sum(v * v, axis=0), 1e-9)
        return expit(mean / np.sqrt(1 + np.pi * var / 8))
