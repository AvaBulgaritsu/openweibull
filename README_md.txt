# Weibull Life-Data Analysis — Open Validation Suite

A reference implementation of two-parameter Weibull life-data analysis with
**correct suspension (right-censoring) handling**, published together with a
suite of falsifiable checks that prove it computes the right answers.

```
python validation_suite.py
```

```
  PASS  Uncensored MLE recovers known parameters (n=200,000)
  PASS  Censored MLE recovers known parameters (40% suspended)
  PASS  Likelihood reproduces the exact exponential MLE (total time / failures)
  PASS  90% intervals actually contain the truth ~90% of the time
  PASS  Two independent methods agree on the same data
  PASS  Correct censoring is accurate; both naive methods are materially biased
  PASS  Flags a two-mode mixture, stays quiet on genuine single-mode data
  PASS  Reported fit is a true likelihood maximum (perturbation check)
  8/8 checks passed
```

---

## Why this exists

Reliability engineers need to answer questions like *"by what mileage will 10%
of these pumps have failed?"* from field data where **most units haven't failed
yet**. Those surviving units are called **suspensions** (or right-censored
observations), and they carry real information: a pump that ran 40,000 miles
without failing tells you something, even though you never saw it fail.

Handling suspensions correctly is the crux of the whole discipline, and it is
exactly what spreadsheets get wrong.

The established tool in this space is trusted largely because it has been
around for decades. A new implementation has no such history. This repository
is the substitute for that history: instead of *"trust us"*, it says
**"here is the arithmetic — reproduce it yourself."**

---

## What the engine does

`weibull_engine.py`

| Function | What it does |
|---|---|
| `weibull_mle(failures, suspensions)` | Maximum-likelihood fit. Solves the profile-likelihood score equation for β by bracketed root-finding, then recovers η in closed form. No starting guess, no convergence babysitting. |
| `median_rank_regression(...)` | The classic probability-plot method, using **Johnson adjusted ranks** for suspensions and Benard's median-rank approximation. Kept as an *independent* estimator — if it disagrees with MLE, something is wrong with the data. |
| `fisher_confidence(...)` | Confidence bounds from the observed information matrix, computed in **log space** so bounds stay positive and small-sample coverage is honest. |
| `b_life(β, η, pct)` | B(pct) life — the time by which pct% of the population has failed. B10 is the industry workhorse. |
| `mttf(β, η)` | Mean time to failure, `η·Γ(1 + 1/β)`. |
| `doglegs_diagnostic(...)` | Detects probable **mixed failure modes** (see below). |
| `naive_drop_suspensions(...)`<br>`naive_treat_as_failures(...)` | The two common spreadsheet mistakes, implemented deliberately so their error can be measured. |

The parameters, in plain terms:

- **β (shape)** — *what kind* of failure. β < 1 means infant mortality
  (failures tail off); β = 1 means random, memoryless; β > 1 means wear-out
  (failures accelerate with age). β tells you the physics.
- **η (scale / characteristic life)** — *when*. By t = η, 63.2% of the
  population has failed, always, for any β.

---

## The eight checks, in plain language

**1 & 2 — Does it find the right answer when we know the right answer?**
Generate 200,000 units from a Weibull with parameters we chose, then ask the
engine to recover them. It gets β and η back to within 0.2%. Test 2 repeats
this with 40% of units suspended, which is the case that actually matters.

**3 — Does it agree with mathematics we can do by hand?**
When β = 1, the Weibull collapses to the exponential distribution, whose MLE
has an exact closed form: *total time on test ÷ number of failures*. On a small
dataset that's `725 ÷ 8 = 90.625`. The engine returns `90.625000`, agreeing to
seven decimal places. This checks the likelihood function itself, including the
suspension term — the suspensions contribute their 300 hours to the numerator,
and if the code dropped them this test would fail loudly.

**4 — Are the error bars honest?**
A "90% confidence interval" that only contains the truth 60% of the time is
worse than no interval at all. This runs 2,000 simulated studies of 30 units
each and counts how often the interval actually captures the true value.
Result: **88.6% for β, 87.9% for η** against a nominal 90%. Slightly
conservative at small sample size, which is the expected and safe direction.

**5 — Do two different methods agree?**
MLE and rank regression share no code and no assumptions beyond the model
itself. On the same data they land within 1.7% of each other. Divergence
between them is a real-world red flag, so the tool computes both.

**6 — The headline: what do the shortcuts actually cost?**
This is the commercial argument, measured rather than asserted. 400 simulated
fleets of 60 units, 45% still running at cutoff:

| Method | β | η | B10 | B10 error |
|---|---|---|---|---|
| **Truth** | 2.00 | 1000 | 325 | — |
| Correct MLE with censoring | 2.10 | 1002 | 337 | **3.9%** |
| Wrong: drop the suspensions | 2.94 | 622 | 287 | 12% *(pessimistic)* |
| Wrong: count them as failures | 3.78 | 785 | 429 | **32% (optimistic)** |

The two mistakes fail in **opposite directions**, and the more dangerous one is
counterintuitive. Dropping suspensions throws away the survivors' good service
time, so the fleet looks worse than it is. Counting suspensions as failures
piles a cluster of fake failures at the cutoff time, which inflates β to 3.8 —
making the distribution look artificially tight — and pushes B10 **32% too
high**. That is an *optimistic* answer: warranty reserves under-funded,
maintenance intervals set too long.

Note also that neither naive β is anywhere near the true 2.0. The inferred
failure *mechanism* is wrong, not just the numbers.

**7 — Does it notice when the model doesn't fit?**
The classic trap is the **"dog-leg"**: two different failure mechanisms in one
dataset (say early manufacturing defects plus late wear-out). Fit one Weibull
through both and you get a confident, plausible, meaningless answer.

Detection uses the probability-plot correlation coefficient with a
**parametric bootstrap** null: fit a single Weibull, simulate 500 datasets of
the same size from that fit, and see whether the observed data is less linear
than a genuine single Weibull plausibly would be.

> **A note on getting this wrong first.** The initial version used a textbook
> Wald–Wolfowitz runs test on the residual signs. It flagged *clean*
> single-mode data at z = −8.55 — a spectacular false positive. The reason is
> that probability-plot points are **order statistics**, so their residuals are
> inherently autocorrelated and the textbook null distribution simply does not
> apply. Calibrating the statistic against its own bootstrap null fixes it.
> This is left documented rather than quietly deleted, because it is exactly
> the class of error the suite exists to catch.

Current behaviour: mixture flagged at p = 0.008, clean data not flagged
(p = 0.588), and a measured **false-positive rate of 5.0%** on 60 clean
samples against a nominal α = 5%.

**8 — Is the reported answer really the best fit?**
Perturb β and η by ±0.01% in all four directions and confirm the likelihood
drops every time. Guards against the optimiser reporting a non-converged point.

---

## Adversarial checks

`stress_test.py` is the separate "try to break it" pass — 21 checks covering
degenerate and extreme inputs:

```
python stress_test.py
```

Covered: complete data with no suspensions; the bare minimum of 2 failures;
tied failure times; the most common real-world pattern (50 units, 4 failures,
46 suspended at one common cutoff); suspensions falling entirely before the
first failure or entirely after the last; scale invariance across 12 orders of
magnitude; 90% censoring; determinism; rejection of invalid input; monotonicity
of B-lives; the definitional identity B63.2 = η; confidence intervals widening
with confidence level; and a proof that the MLE really does beat rank
regression on likelihood. It also re-derives every number quoted in this README
and fails if any has drifted.

---

## Worked example you can check by hand

```
8 failures:     31, 39, 57, 65, 72, 89, 104, 128
6 suspensions:  45, 45, 80, 110, 110, 110

β    = 2.5328     90% CI (1.5670, 4.0937)
η    = 107.13     90% CI (84.87, 135.24)
B10  = 44.06      MTTF = 95.09
log-likelihood = -43.806634
```

β ≈ 2.5 means wear-out, not random failure — the units are ageing, so
preventive replacement makes sense. B10 ≈ 44 means 10% will have failed by 44
hours. Note the wide β interval: with only 8 failures, the data genuinely
cannot pin the mechanism down tightly, and the tool says so instead of
projecting false precision.

---

## Scope and limits

Deliberately narrow. It does **not** yet do: 3-parameter Weibull (failure-free
period), interval/left censoring, competing-risk decomposition, Bayesian
priors, accelerated life testing, or repairable-system models. It does the
core case — 2-parameter Weibull, complete or right-censored — and proves it.

Requires `numpy` and `scipy`.

---

## Reuse

MIT. Copy it, check it, argue with it. If any check here fails on your machine
or you can construct a case where the engine is wrong, that is the most useful
possible contribution.
