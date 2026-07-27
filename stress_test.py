"""
stress_test.py — adversarial checks before publishing.

Not part of the validation story. This is the "what would an expert try to
break" pass: edge cases, degenerate inputs, and numerical extremes.
"""
import numpy as np
from scipy.stats import weibull_min
from weibull_engine import (weibull_mle, median_rank_regression,
                            fisher_confidence, b_life, log_likelihood,
                            doglegs_diagnostic)

fails = []
def chk(name, cond, note=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  — {note}" if note else ""))
    if not cond:
        fails.append(name)

print("EDGE CASES\n")

# 1. Complete data, no suspensions
b, e = weibull_mle([10, 20, 30, 40, 50])
chk("complete data (no suspensions)", np.isfinite(b) and np.isfinite(e), f"beta={b:.3f} eta={e:.2f}")

# 2. Absolute minimum: 2 failures
b, e = weibull_mle([10, 20])
chk("minimum 2 failures", np.isfinite(b) and np.isfinite(e), f"beta={b:.3f} eta={e:.2f}")

# 3. Ties in failure times (very common in real logs)
b, e = weibull_mle([10, 10, 10, 20, 20, 30], [30, 30])
chk("tied failure times", np.isfinite(b) and b > 0, f"beta={b:.3f} eta={e:.2f}")

# 4. All suspensions at one common cutoff — THE most common real pattern
f = [12, 25, 31, 44]
s = [60] * 46
b, e = weibull_mle(f, s)
chk("50 units, 4 failures, 46 suspended at common cutoff",
    np.isfinite(b) and np.isfinite(e), f"beta={b:.3f} eta={e:.2f}")

# 5. Suspensions all BEFORE the first failure
b, e = weibull_mle([50, 60, 70], [5, 6, 7])
chk("suspensions before first failure", np.isfinite(b) and b > 0, f"beta={b:.3f} eta={e:.2f}")

# 6. Suspensions all AFTER the last failure
b, e = weibull_mle([10, 20, 30], [100, 100, 100])
chk("suspensions after last failure", np.isfinite(b) and b > 0, f"beta={b:.3f} eta={e:.2f}")

# 7. Extreme magnitudes — scale invariance check
base_f, base_s = [10., 20., 30., 40.], [50., 50.]
b1, e1 = weibull_mle(base_f, base_s)
for k in (1e-6, 1e6):
    b2, e2 = weibull_mle([x * k for x in base_f], [x * k for x in base_s])
    ok = abs(b2 - b1) / b1 < 1e-6 and abs(e2 / k - e1) / e1 < 1e-6
    chk(f"scale invariance x{k:g}", ok, f"beta {b1:.6f} -> {b2:.6f}")

# 8. Heavy censoring: 90% suspended
rng = np.random.default_rng(7)
t = weibull_min.rvs(2.0, scale=100.0, size=200, random_state=rng)
cut = np.quantile(t, 0.10)
f, s = t[t <= cut], np.full(int((t > cut).sum()), cut)
b, e = weibull_mle(f, s)
chk("90% censored", np.isfinite(b) and 0.5 < b < 6, f"beta={b:.3f} (true 2.0), n_fail={len(f)}")

# 9. Determinism — same input twice
r1 = weibull_mle([31, 39, 57, 65], [45, 80])
r2 = weibull_mle([31, 39, 57, 65], [45, 80])
chk("deterministic", r1 == r2)

# 10. Rejects bad input rather than returning nonsense
bad = 0
for args in ([[10]], [[10, -5]], [[10, 20], [0]]):
    try:
        weibull_mle(*args)
    except ValueError:
        bad += 1
chk("rejects invalid input (1 failure / negative / zero)", bad == 3, f"{bad}/3 raised")

print("\nCONSISTENCY OF DERIVED QUANTITIES\n")

# 11. B-lives must be monotone increasing
b, e = weibull_mle([31, 39, 57, 65, 72, 89, 104, 128], [45, 45, 80, 110, 110, 110])
bl = [b_life(b, e, p) for p in (1, 5, 10, 25, 50, 63.2, 90)]
chk("B-lives monotone increasing", all(x < y for x, y in zip(bl, bl[1:])),
    " < ".join(f"{x:.1f}" for x in bl))

# 12. B63.2 must equal eta (definitional)
chk("B63.2 == eta", abs(b_life(b, e, 63.212) - e) / e < 1e-3,
    f"B63.2={b_life(b, e, 63.212):.4f} vs eta={e:.4f}")

# 13. CI must bracket the point estimate
ci = fisher_confidence([31, 39, 57, 65, 72, 89, 104, 128], [45, 45, 80, 110, 110, 110])
ok = ci["beta_ci"][0] < ci["beta"] < ci["beta_ci"][1] and ci["eta_ci"][0] < ci["eta"] < ci["eta_ci"][1]
chk("CI brackets point estimate", ok)

# 14. Wider confidence => wider interval
c50 = fisher_confidence([31, 39, 57, 65], [45, 80], conf=0.50)
c99 = fisher_confidence([31, 39, 57, 65], [45, 80], conf=0.99)
w50 = c50["beta_ci"][1] - c50["beta_ci"][0]
w99 = c99["beta_ci"][1] - c99["beta_ci"][0]
chk("99% interval wider than 50%", w99 > w50, f"{w50:.3f} vs {w99:.3f}")

# 15. MLE beats rank regression on likelihood (it must, by definition)
f2 = [31, 39, 57, 65, 72, 89, 104, 128]; s2 = [45, 45, 80, 110, 110, 110]
bm, em = weibull_mle(f2, s2)
br, er, _ = median_rank_regression(f2, s2)
chk("MLE has higher likelihood than rank regression",
    log_likelihood(bm, em, f2, s2) >= log_likelihood(br, er, f2, s2),
    f"{log_likelihood(bm, em, f2, s2):.4f} >= {log_likelihood(br, er, f2, s2):.4f}")

print("\nREADME CLAIMS\n")

# 16. The published worked example must reproduce exactly
bw, ew = weibull_mle(f2, s2)
ciw = fisher_confidence(f2, s2)
claims = [
    ("beta = 2.5328", abs(bw - 2.5328) < 5e-4),
    ("eta = 107.13", abs(ew - 107.13) < 5e-3),
    ("B10 = 44.06", abs(b_life(bw, ew, 10) - 44.06) < 5e-3),
    ("beta CI (1.5670, 4.0937)", abs(ciw["beta_ci"][0] - 1.5670) < 5e-4
                                 and abs(ciw["beta_ci"][1] - 4.0937) < 5e-4),
    ("logL = -43.806634", abs(log_likelihood(bw, ew, f2, s2) + 43.806634) < 1e-5),
]
for label, ok in claims:
    chk(f"README worked example: {label}", ok)

print("\nKNOWN LIMITATIONS (documented, not bugs)\n")
print("  * Fisher/normal-approximation bounds. Likelihood-ratio bounds are")
print("    preferred by many practitioners below ~20 failures; ours are")
print("    slightly conservative there (measured 88.6% vs nominal 90%).")
print("  * 2-parameter Weibull only. No location parameter (gamma), so data")
print("    with a genuine failure-free period will fit poorly.")
print("  * Right-censoring only. No interval or left censoring.")
print("  * Dog-leg detector flags THAT a mixture is likely, not which modes.")

print("\n" + "=" * 60)
print(f"  {'ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILURES: ' + ', '.join(fails)}")
print("=" * 60)
raise SystemExit(1 if fails else 0)
