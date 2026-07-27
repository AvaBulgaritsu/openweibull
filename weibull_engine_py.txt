"""
weibull_engine.py — Life data analysis for reliability engineering.

Two-parameter Weibull fitting that handles SUSPENSIONS (right-censored units)
correctly. Suspensions are units that had not failed when observation stopped.
Mishandling them is the single most common error in spreadsheet-based analysis,
and it biases every downstream number: B10 life, MTTF, warranty forecasts.

Conventions
-----------
beta  (b) : shape parameter. beta<1 infant mortality, beta=1 random,
            beta>1 wear-out.
eta   (n) : scale parameter (characteristic life). F(eta) = 63.2%.

Reliability function : R(t) = exp(-(t/eta)**beta)
CDF (unreliability)  : F(t) = 1 - R(t)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# --------------------------------------------------------------------------
# Core likelihood
# --------------------------------------------------------------------------

def log_likelihood(beta: float, eta: float,
                   failures: np.ndarray,
                   suspensions: np.ndarray) -> float:
    """Exact log-likelihood for a 2-parameter Weibull with right-censoring.

    Each FAILURE contributes its pdf; each SUSPENSION contributes its
    survival probability. Dropping the suspension term (or, worse, treating
    suspensions as failures) is exactly the spreadsheet mistake.
    """
    if beta <= 0 or eta <= 0:
        return -np.inf

    f = np.asarray(failures, dtype=float)
    s = np.asarray(suspensions, dtype=float)
    r = f.size

    ll = 0.0
    if r:
        ll += r * np.log(beta) - r * beta * np.log(eta)
        ll += (beta - 1.0) * np.sum(np.log(f))
        ll -= np.sum((f / eta) ** beta)
    if s.size:
        # survival contribution of censored units
        ll -= np.sum((s / eta) ** beta)
    return float(ll)


def _beta_score(beta: float, f: np.ndarray, allt: np.ndarray, r: int) -> float:
    """Profile-likelihood score equation for beta.

    Setting d(logL)/d(beta) = 0 after substituting the closed-form eta gives:

        sum(t^b * ln t) / sum(t^b)  -  1/b  -  (1/r) * sum_failures(ln t) = 0

    where the t-sums run over ALL units (failures + suspensions) and the
    final term runs over failures only.
    """
    tb = allt ** beta
    num = np.sum(tb * np.log(allt))
    den = np.sum(tb)
    return num / den - 1.0 / beta - np.mean(np.log(f))


def weibull_mle(failures, suspensions=(), return_ll: bool = False):
    """Maximum-likelihood fit of a 2-parameter Weibull with suspensions.

    Solves the profile-likelihood score equation for beta by bracketed
    root-finding (robust, no starting guess needed), then recovers eta in
    closed form:  eta = ( sum(t^beta) / r ) ** (1/beta).

    Parameters
    ----------
    failures    : observed failure times (>0), at least 2 required.
    suspensions : right-censored times (>0). May be empty.

    Returns
    -------
    (beta, eta) or (beta, eta, loglik) if return_ll.
    """
    f = np.asarray(failures, dtype=float)
    s = np.asarray(suspensions, dtype=float)

    if f.size < 2:
        raise ValueError("need at least 2 failures to fit 2 parameters")
    if np.any(f <= 0) or np.any(s <= 0):
        raise ValueError("all times must be strictly positive")

    allt = np.concatenate([f, s]) if s.size else f
    r = f.size

    # Bracket the root. The score is monotone INCREASING in beta:
    #   beta -> 0+   the -1/beta term dominates  => score -> -inf
    #   beta -> inf  the t-sum tends to ln(t_max) => score -> ln(t_max) - mean(ln f) > 0
    # So expand until score(lo) < 0 < score(hi).
    lo, hi = 1e-3, 1.0
    while _beta_score(hi, f, allt, r) < 0 and hi < 1e6:
        hi *= 2.0
    while _beta_score(lo, f, allt, r) > 0 and lo > 1e-10:
        lo /= 2.0

    beta = brentq(_beta_score, lo, hi, args=(f, allt, r), xtol=1e-12, rtol=1e-14)
    eta = (np.sum(allt ** beta) / r) ** (1.0 / beta)

    if return_ll:
        return beta, eta, log_likelihood(beta, eta, f, s)
    return beta, eta


# --------------------------------------------------------------------------
# Median rank regression (the graphical method, with Johnson suspension ranks)
# --------------------------------------------------------------------------

def median_rank_regression(failures, suspensions=()):
    """Rank-regression fit using Johnson adjusted ranks + Benard's approximation.

    This is the classic 'probability plot' method. It is included as an
    INDEPENDENT estimator: if MLE and MRR disagree wildly on the same data,
    something is wrong with the data or the model, and the tool should say so
    rather than silently reporting one number.

    Adjusted rank increment for each failure:
        I = (N + 1 - prev_adjusted_rank) / (1 + n_beyond)
    Benard's median rank:
        MR = (AR - 0.3) / (N + 0.4)
    """
    f = np.asarray(failures, dtype=float)
    s = np.asarray(suspensions, dtype=float)
    N = f.size + s.size
    if f.size < 2:
        raise ValueError("need at least 2 failures")

    # Build ordered event list: (time, is_failure). Suspensions at equal time
    # are treated as occurring after failures (standard convention).
    events = [(t, 1) for t in f] + [(t, 0) for t in s]
    events.sort(key=lambda x: (x[0], -x[1]))

    prev_ar = 0.0
    ranks, times = [], []
    for idx, (t, is_fail) in enumerate(events):
        n_beyond = N - idx          # units remaining including this one
        if is_fail:
            inc = (N + 1.0 - prev_ar) / (1.0 + n_beyond)
            ar = prev_ar + inc
            prev_ar = ar
            ranks.append(ar)
            times.append(t)

    ar = np.array(ranks)
    tt = np.array(times)
    mr = (ar - 0.3) / (N + 0.4)

    # Linearise: ln(-ln(1-F)) = beta*ln(t) - beta*ln(eta)
    y = np.log(-np.log(1.0 - mr))
    x = np.log(tt)

    # Reliability convention: regress X on Y (rank regression on X)
    slope_xy, intercept_xy = np.polyfit(y, x, 1)
    beta = 1.0 / slope_xy
    eta = np.exp(intercept_xy)

    # Pearson correlation of the probability plot — a linearity diagnostic
    rho = float(np.corrcoef(x, y)[0, 1])
    return beta, eta, rho


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def _num_hessian(fun, p, h=1e-5):
    """Central-difference Hessian of a scalar function at p."""
    p = np.asarray(p, dtype=float)
    n = p.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pp = p.copy(); pp[i] += h; pp[j] += h; a = fun(pp)
            pm = p.copy(); pm[i] += h; pm[j] -= h; b = fun(pm)
            mp = p.copy(); mp[i] -= h; mp[j] += h; c = fun(mp)
            mm = p.copy(); mm[i] -= h; mm[j] -= h; d = fun(mm)
            H[i, j] = (a - b - c + d) / (4 * h * h)
    return H


def fisher_confidence(failures, suspensions=(), conf=0.90):
    """Two-sided confidence bounds on beta and eta via the observed
    information matrix, computed in LOG space.

    Log-transforming keeps the bounds strictly positive and gives noticeably
    better small-sample coverage than naive symmetric bounds — which matters,
    because reliability datasets are usually small.
    """
    beta, eta = weibull_mle(failures, suspensions)

    def nll_log(p):
        return -log_likelihood(np.exp(p[0]), np.exp(p[1]), failures, suspensions)

    H = _num_hessian(nll_log, [np.log(beta), np.log(eta)])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return dict(beta=beta, eta=eta, beta_ci=(np.nan, np.nan),
                    eta_ci=(np.nan, np.nan), ok=False)

    if np.any(np.diag(cov) < 0):
        return dict(beta=beta, eta=eta, beta_ci=(np.nan, np.nan),
                    eta_ci=(np.nan, np.nan), ok=False)

    z = norm.ppf(0.5 + conf / 2.0)
    sb, se = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return dict(
        beta=beta, eta=eta, ok=True,
        beta_ci=(beta * np.exp(-z * sb), beta * np.exp(z * sb)),
        eta_ci=(eta * np.exp(-z * se), eta * np.exp(z * se)),
    )


# --------------------------------------------------------------------------
# Derived engineering quantities
# --------------------------------------------------------------------------

def b_life(beta: float, eta: float, pct: float) -> float:
    """B(pct) life — the time by which `pct` percent of the population fails.
    B10 is the industry workhorse."""
    p = pct / 100.0
    return eta * (-np.log(1.0 - p)) ** (1.0 / beta)


def mttf(beta: float, eta: float) -> float:
    """Mean time to failure = eta * Gamma(1 + 1/beta)."""
    from scipy.special import gamma
    return eta * gamma(1.0 + 1.0 / beta)


def reliability(t, beta: float, eta: float):
    return np.exp(-(np.asarray(t, dtype=float) / eta) ** beta)


# --------------------------------------------------------------------------
# The naive estimators — implemented deliberately, to quantify their error
# --------------------------------------------------------------------------

def naive_drop_suspensions(failures, suspensions=()):
    """WRONG METHOD 1: throw the suspensions away and fit only failures.

    This is what happens when someone pastes 'the failures' into a
    spreadsheet. It discards the information that the surviving units
    lasted at least as long as they did, and biases eta DOWNWARD —
    the fleet looks worse than it is.
    """
    return weibull_mle(failures, ())


def naive_treat_as_failures(failures, suspensions=()):
    """WRONG METHOD 2: treat suspensions as if they were failures.

    Common when a maintenance log doesn't distinguish 'removed' from
    'failed'. Biases eta DOWNWARD hard and distorts beta — the fleet looks
    far worse than it is.
    """
    allt = np.concatenate([np.asarray(failures, float),
                           np.asarray(suspensions, float)])
    return weibull_mle(allt, ())


# --------------------------------------------------------------------------
# Diagnostics — the "don't confidently report a wrong number" layer
# --------------------------------------------------------------------------

def doglegs_diagnostic(failures, suspensions=(), n_boot: int = 500, alpha: float = 0.05,
                       seed: int = 12345):
    """Detect probable MIXED FAILURE MODES ('dog-leg' on the probability plot).

    A single Weibull fitted through two overlapping failure mechanisms gives a
    confident, plausible, and wrong answer. Classic reliability literature warns
    about it constantly; naive tools fit straight through it and say nothing.

    Method: probability-plot correlation coefficient (PPCC) with a PARAMETRIC
    BOOTSTRAP null.

    Why a bootstrap rather than a fixed rho cut-off or a textbook runs test:
    the points on a probability plot are ORDER STATISTICS, so their residuals
    are inherently autocorrelated. A naive Wald-Wolfowitz runs test therefore
    fires on perfectly good data (we tried; it flagged clean single-mode
    samples at z = -8). The honest fix is to calibrate the statistic against
    its own null distribution: fit a single Weibull, simulate many datasets of
    the same size from that fit, and see where the observed rho falls.

      p = P( rho_simulated <= rho_observed | data really is one Weibull )

    Small p means the data is less linear than a genuine single Weibull would
    plausibly be => suspect a mixture.

    Returns statistics plus a boolean `suspect_mixture`.
    """
    from scipy.stats import weibull_min

    beta, eta, rho_obs = median_rank_regression(failures, suspensions)

    n_f = len(failures)
    n_s = len(suspensions) if suspensions is not None else 0
    rng = np.random.default_rng(seed)

    # Null distribution of rho under "it really is a single Weibull".
    # Reproduce the same failure/suspension counts so the comparison is fair.
    boot = []
    for _ in range(n_boot):
        t = weibull_min.rvs(beta, scale=eta, size=n_f + n_s, random_state=rng)
        t.sort()
        if n_s:
            sim_f, sim_s = t[:n_f], t[n_f:]
        else:
            sim_f, sim_s = t, np.array([])
        try:
            _, _, r = median_rank_regression(sim_f, sim_s)
            boot.append(r)
        except Exception:
            continue

    boot = np.asarray(boot)
    p_value = float(np.mean(boot <= rho_obs)) if boot.size else float("nan")
    rho_crit = float(np.quantile(boot, alpha)) if boot.size else float("nan")

    return dict(
        rho=float(rho_obs),
        rho_critical=rho_crit,       # below this => unusually non-linear
        p_value=p_value,
        n_boot=int(boot.size),
        suspect_mixture=bool(p_value < alpha),
        beta_single=float(beta),
        eta_single=float(eta),
    )
