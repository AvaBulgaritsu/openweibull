"""
validation_suite.py — Public evidence that the engine computes the right answer.

Every test here is FALSIFIABLE and self-contained: it either passes against a
ground truth we construct, or it fails loudly. Nothing is asserted on
authority. Run `python validation_suite.py` and read the report.

Why this exists
---------------
A newcomer selling reliability software has no brand to lean on. The incumbent
tool's real moat is thirty years of people trusting its numbers. This suite is
the substitute for those thirty years: instead of "trust us", it says
"here is the arithmetic, reproduce it yourself."
"""

from __future__ import annotations

import numpy as np
from scipy.stats import weibull_min

from weibull_engine import (
    weibull_mle, median_rank_regression, fisher_confidence,
    log_likelihood, b_life, mttf,
    naive_drop_suspensions, naive_treat_as_failures,
    doglegs_diagnostic,
)

RNG = np.random.default_rng(20260727)
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
    for line in detail.strip().splitlines():
        print(f"          {line}")
    print()


def gen_censored(beta, eta, n, censor_frac, rng):
    """Generate a realistic right-censored sample: units observed until a
    common cutoff time, so survivors become suspensions."""
    t = weibull_min.rvs(beta, scale=eta, size=n, random_state=rng)
    cutoff = np.quantile(t, 1.0 - censor_frac)
    failures = t[t <= cutoff]
    suspensions = np.full(int(np.sum(t > cutoff)), cutoff)
    return failures, suspensions


# ---------------------------------------------------------------------------
# TEST 1 — Consistency: with lots of data the estimator must find the truth
# ---------------------------------------------------------------------------
def test_parameter_recovery():
    print("TEST 1 — Parameter recovery (estimator consistency)")
    true_beta, true_eta, n = 2.5, 1000.0, 200_000
    t = weibull_min.rvs(true_beta, scale=true_eta, size=n, random_state=RNG)
    beta, eta = weibull_mle(t)
    eb = abs(beta - true_beta) / true_beta * 100
    ee = abs(eta - true_eta) / true_eta * 100
    record(
        "Uncensored MLE recovers known parameters (n=200,000)",
        eb < 1.0 and ee < 1.0,
        f"true  beta={true_beta:.4f}  eta={true_eta:.1f}\n"
        f"fit   beta={beta:.4f}  eta={eta:.1f}\n"
        f"error beta={eb:.3f}%   eta={ee:.3f}%   (tolerance 1%)",
    )


# ---------------------------------------------------------------------------
# TEST 2 — Censored consistency: suspensions handled right => still consistent
# ---------------------------------------------------------------------------
def test_censored_recovery():
    print("TEST 2 — Parameter recovery WITH suspensions")
    true_beta, true_eta = 1.8, 500.0
    failures, susp = gen_censored(true_beta, true_eta, 200_000, 0.40, RNG)
    beta, eta = weibull_mle(failures, susp)
    eb = abs(beta - true_beta) / true_beta * 100
    ee = abs(eta - true_eta) / true_eta * 100
    record(
        "Censored MLE recovers known parameters (40% suspended)",
        eb < 1.5 and ee < 1.5,
        f"n=200,000 units, {len(failures):,} failures, {len(susp):,} suspensions\n"
        f"true  beta={true_beta:.4f}  eta={true_eta:.1f}\n"
        f"fit   beta={beta:.4f}  eta={eta:.1f}\n"
        f"error beta={eb:.3f}%   eta={ee:.3f}%   (tolerance 1.5%)",
    )


# ---------------------------------------------------------------------------
# TEST 3 — Analytic check: beta fixed at 1 => Weibull is Exponential
# ---------------------------------------------------------------------------
def test_exponential_closed_form():
    print("TEST 3 — Closed-form agreement (exponential special case)")
    # For beta = 1 the MLE of eta has the exact closed form:
    #     eta_hat = total time on test / number of failures
    failures = np.array([12., 25., 31., 44., 58., 71., 88., 96.])
    susp = np.array([100., 100., 100.])
    total_time = failures.sum() + susp.sum()
    analytic_eta = total_time / failures.size

    # Constrain the engine's likelihood to beta=1 and maximise over eta only.
    from scipy.optimize import minimize_scalar
    obj = lambda le: -log_likelihood(1.0, np.exp(le), failures, susp)
    num_eta = np.exp(minimize_scalar(obj, bracket=(2.0, 6.0)).x)

    err = abs(num_eta - analytic_eta) / analytic_eta * 100
    record(
        "Likelihood reproduces the exact exponential MLE (total time / failures)",
        err < 1e-4,
        f"analytic eta = {total_time:.1f} / {failures.size} = {analytic_eta:.6f}\n"
        f"engine   eta = {num_eta:.6f}\n"
        f"relative error = {err:.2e}%   (tolerance 1e-4%)",
    )


# ---------------------------------------------------------------------------
# TEST 4 — Confidence interval COVERAGE (a real Monte Carlo study)
# ---------------------------------------------------------------------------
def test_ci_coverage():
    print("TEST 4 — Confidence-interval coverage (Monte Carlo, 2,000 trials)")
    true_beta, true_eta, n, trials, conf = 2.0, 300.0, 30, 2000, 0.90
    hit_b = hit_e = valid = 0
    for _ in range(trials):
        f, s = gen_censored(true_beta, true_eta, n, 0.30, RNG)
        if f.size < 5:
            continue
        try:
            r = fisher_confidence(f, s, conf=conf)
        except Exception:
            continue
        if not r["ok"]:
            continue
        valid += 1
        if r["beta_ci"][0] <= true_beta <= r["beta_ci"][1]:
            hit_b += 1
        if r["eta_ci"][0] <= true_eta <= r["eta_ci"][1]:
            hit_e += 1
    cb, ce = hit_b / valid, hit_e / valid
    record(
        "90% intervals actually contain the truth ~90% of the time",
        0.85 <= cb <= 0.95 and 0.85 <= ce <= 0.95,
        f"n=30 units per trial, 30% suspended, {valid} valid trials\n"
        f"beta coverage = {cb*100:.1f}%   (nominal 90%)\n"
        f"eta  coverage = {ce*100:.1f}%   (nominal 90%)\n"
        f"acceptance band 85-95%",
    )


# ---------------------------------------------------------------------------
# TEST 5 — Cross-method agreement: MLE vs rank regression
# ---------------------------------------------------------------------------
def test_cross_method():
    print("TEST 5 — Independent estimator agreement (MLE vs median rank regression)")
    true_beta, true_eta = 2.2, 800.0
    f, s = gen_censored(true_beta, true_eta, 5000, 0.25, RNG)
    b1, e1 = weibull_mle(f, s)
    b2, e2, rho = median_rank_regression(f, s)
    db = abs(b1 - b2) / b1 * 100
    de = abs(e1 - e2) / e1 * 100
    record(
        "Two independent methods agree on the same data",
        db < 5.0 and de < 5.0 and rho > 0.99,
        f"MLE          beta={b1:.4f}  eta={e1:.2f}\n"
        f"rank regr.   beta={b2:.4f}  eta={e2:.2f}\n"
        f"difference   beta={db:.2f}%    eta={de:.2f}%   (tolerance 5%)\n"
        f"probability-plot correlation rho={rho:.5f}",
    )


# ---------------------------------------------------------------------------
# TEST 6 — THE HEADLINE: quantify the cost of mishandling suspensions
# ---------------------------------------------------------------------------
def test_suspension_bias():
    print("TEST 6 — Bias of the two common spreadsheet mistakes  [HEADLINE]")
    true_beta, true_eta = 2.0, 1000.0
    n, trials = 60, 400
    correct, drop, asfail = [], [], []
    for _ in range(trials):
        f, s = gen_censored(true_beta, true_eta, n, 0.45, RNG)
        if f.size < 5:
            continue
        correct.append(weibull_mle(f, s))
        drop.append(naive_drop_suspensions(f, s))
        asfail.append(naive_treat_as_failures(f, s))

    correct, drop, asfail = map(np.array, (correct, drop, asfail))
    true_b10 = b_life(true_beta, true_eta, 10)

    def summarise(arr):
        b, e = arr[:, 0].mean(), arr[:, 1].mean()
        b10 = np.mean([b_life(x, y, 10) for x, y in arr])
        return b, e, b10

    cb, ce, cb10 = summarise(correct)
    db, de, db10 = summarise(drop)
    ab, ae, ab10 = summarise(asfail)

    err_correct = abs(cb10 - true_b10) / true_b10 * 100
    err_drop = abs(db10 - true_b10) / true_b10 * 100
    err_asfail = abs(ab10 - true_b10) / true_b10 * 100

    record(
        "Correct censoring is accurate; both naive methods are materially biased",
        err_correct < 8.0 and err_drop > 8.0 and err_asfail > 20.0,
        f"{trials} simulated fleets, n={n} units, 45% still running at cutoff\n"
        f"TRUTH                     beta={true_beta:.3f}  eta={true_eta:7.1f}  B10={true_b10:7.1f}\n"
        f"correct MLE (censored)    beta={cb:.3f}  eta={ce:7.1f}  B10={cb10:7.1f}"
        f"   B10 error {err_correct:5.1f}%\n"
        f"WRONG: drop suspensions   beta={db:.3f}  eta={de:7.1f}  B10={db10:7.1f}"
        f"   B10 error {err_drop:5.1f}%  (pessimistic)\n"
        f"WRONG: count as failures  beta={ab:.3f}  eta={ae:7.1f}  B10={ab10:7.1f}"
        f"   B10 error {err_asfail:5.1f}%  (OPTIMISTIC)\n"
        f"\nReading: the two shortcuts fail in OPPOSITE directions, and the\n"
        f"more common one is the dangerous one. Dropping suspensions throws\n"
        f"away the survivors' good service time, so the fleet looks worse\n"
        f"(B10 low by {err_drop:.0f}%). Counting suspensions as failures piles a\n"
        f"cluster of fake failures at the cutoff, which inflates beta to\n"
        f"{ab:.1f} and makes the distribution look artificially tight — so B10\n"
        f"comes out {err_asfail:.0f}% HIGH. That is an optimistic answer: warranty\n"
        f"reserves and maintenance intervals both get set too loose.\n"
        f"Note neither naive beta is near the truth ({true_beta:.1f}), so the\n"
        f"inferred failure MECHANISM is wrong too, not just the numbers.",
    )


# ---------------------------------------------------------------------------
# TEST 7 — Diagnostics catch a mixed failure mode ("dog-leg")
# ---------------------------------------------------------------------------
def test_dogleg_detection():
    print("TEST 7 — Mixed-failure-mode ('dog-leg') detection")
    # Population A: early infant mortality. Population B: late wear-out.
    a = weibull_min.rvs(0.8, scale=120.0, size=45, random_state=RNG)
    b = weibull_min.rvs(4.5, scale=1400.0, size=75, random_state=RNG)
    mixed = np.concatenate([a, b])
    clean = weibull_min.rvs(2.3, scale=900.0, size=120, random_state=RNG)

    dm = doglegs_diagnostic(mixed)
    dc = doglegs_diagnostic(clean)

    # False-positive rate: the flag must stay quiet on genuinely clean data.
    fp = 0
    for _ in range(60):
        g = weibull_min.rvs(2.0, scale=700.0, size=100, random_state=RNG)
        if doglegs_diagnostic(g, n_boot=200)["suspect_mixture"]:
            fp += 1
    fp_rate = fp / 60.0

    record(
        "Flags a two-mode mixture, stays quiet on genuine single-mode data",
        dm["suspect_mixture"] and not dc["suspect_mixture"] and fp_rate <= 0.12,
        f"MIXED (infant mortality beta=0.8 + wear-out beta=4.5, 45+75 units)\n"
        f"   rho={dm['rho']:.4f}  vs critical {dm['rho_critical']:.4f}"
        f"   p={dm['p_value']:.4f}  -> flagged: {dm['suspect_mixture']}\n"
        f"   a naive single fit would report beta={dm['beta_single']:.3f}, "
        f"eta={dm['eta_single']:.1f}\n"
        f"   — that beta is a meaningless blend of two mechanisms, and it is\n"
        f"     the number an unguarded tool would print with no warning.\n"
        f"SINGLE MODE (120 units, genuine Weibull)\n"
        f"   rho={dc['rho']:.4f}  vs critical {dc['rho_critical']:.4f}"
        f"   p={dc['p_value']:.4f}  -> flagged: {dc['suspect_mixture']}\n"
        f"FALSE-POSITIVE RATE on 60 clean samples: {fp_rate*100:.1f}%"
        f"   (nominal alpha 5%, acceptance <=12%)",
    )


# ---------------------------------------------------------------------------
# TEST 8 — Worked example, fully auditable by hand
# ---------------------------------------------------------------------------
def test_worked_example():
    print("TEST 8 — Auditable worked example (small dataset, printable)")
    failures = np.array([31., 39., 57., 65., 72., 89., 104., 128.])
    susp = np.array([45., 45., 80., 110., 110., 110.])

    beta, eta = weibull_mle(failures, susp)
    ci = fisher_confidence(failures, susp, conf=0.90)
    ll = log_likelihood(beta, eta, failures, susp)

    # Independent verification: the MLE must be a stationary point of the
    # likelihood. Perturb each parameter and confirm the likelihood drops.
    d = 1e-4
    worse = (log_likelihood(beta * (1 + d), eta, failures, susp) < ll and
             log_likelihood(beta * (1 - d), eta, failures, susp) < ll and
             log_likelihood(beta, eta * (1 + d), failures, susp) < ll and
             log_likelihood(beta, eta * (1 - d), failures, susp) < ll)

    record(
        "Reported fit is a true likelihood maximum (perturbation check)",
        worse,
        f"8 failures: {failures.astype(int).tolist()}\n"
        f"6 suspensions: {susp.astype(int).tolist()}\n"
        f"beta = {beta:.4f}   90% CI ({ci['beta_ci'][0]:.4f}, {ci['beta_ci'][1]:.4f})\n"
        f"eta  = {eta:.2f}   90% CI ({ci['eta_ci'][0]:.2f}, {ci['eta_ci'][1]:.2f})\n"
        f"B10  = {b_life(beta, eta, 10):.2f}     MTTF = {mttf(beta, eta):.2f}\n"
        f"log-likelihood = {ll:.6f}\n"
        f"perturbing either parameter by +/-0.01% lowers the likelihood: {worse}",
    )


def main():
    print("=" * 78)
    print("WEIBULL LIFE-DATA ANALYSIS — VALIDATION SUITE")
    print("Reproducible evidence that the engine computes correct answers.")
    print("=" * 78 + "\n")

    test_parameter_recovery()
    test_censored_recovery()
    test_exponential_closed_form()
    test_ci_coverage()
    test_cross_method()
    test_suspension_bias()
    test_dogleg_detection()
    test_worked_example()

    print("=" * 78)
    passed = sum(1 for _, p, _ in RESULTS if p)
    for name, p, _ in RESULTS:
        print(f"  {'PASS' if p else 'FAIL'}  {name}")
    print("-" * 78)
    print(f"  {passed}/{len(RESULTS)} checks passed")
    print("=" * 78)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
