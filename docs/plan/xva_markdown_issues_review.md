# Issues Identified in Original `xva_mathematics.md`

This document lists **only concrete mathematical or specification issues** in the original markdown, with precise explanations of what is wrong and why.

---

## 1️⃣ Marginal Default Probability Formula (Incorrect Equality)

### Location
Section defining:
```
ΔPD(t_{i-1}, t_i)= S(t_{i-1}) - S(t_i) = 1 - S(t_i)/S(t_{i-1})
```

### What is wrong
The two expressions are **not equal in general**.

- Unconditional bucket default probability:
  ΔPD = S(t_{i-1}) − S(t_i)

- Conditional default probability given survival to t_{i-1}:
  ΔPD_cond = 1 − S(t_i)/S(t_{i-1})

These are related by:
  S(t_{i-1}) − S(t_i) = S(t_{i-1}) × (1 − S(t_i)/S(t_{i-1}))

The second expression must be multiplied by S(t_{i-1}) to equal the first.

### Why this matters
CVA uses **unconditional** default probabilities in the bucketed sum. Using the conditional expression directly would produce incorrect CVA.

---

## 2️⃣ Hazard Rate Extraction is Underspecified

### Location
Section stating hazard rate h(t) is "extracted numerically from survival pillars".

### What is wrong
The method is not defined.

Possible approaches produce different results:
- Piecewise-constant hazard
- Linear interpolation of hazard
- Log-linear interpolation of survival
- Numerical differentiation

### Why this matters
Different interpolation methods produce different bucket default probabilities and therefore different CVA/DVA.

Recommended specification:
Use piecewise-constant hazard implied by:
  h_k = [ln S(t_{k-1}) − ln S(t_k)] / (t_k − t_{k-1})

or avoid hazard entirely and compute ΔPD directly from interpolated survival.

---

## 3️⃣ Credit Curve Interpolation Not Fully Specified

### Location
Survival curve section.

### What is wrong
Interpolation method for S(t) is not clearly fixed.

Without explicitly enforcing monotonicity:
- S(t) could become non-monotone
- Hazard rate could become negative

### Why this matters
Survival must be:
- S(0)=1
- Non-increasing
- In [0,1]

Recommendation:
Use log-linear interpolation on survival (linear in log S).

---

## 4️⃣ Collateral Formula Lacks Explicit Sign Convention

### Location
Collateral formula:
```
C(t,ω)=max(V_NS−H,0)−max(−V_NS−H,0)
```

### What is wrong
The sign convention for:
- V_NS
- C
- Exposure definitions

is not explicitly defined.

### Why this matters
If C is signed (positive = collateral held, negative = posted), then exposure formulas must match that convention exactly. Without stating the sign framework explicitly, implementation ambiguity arises.

---

## 5️⃣ IRS Float Leg Under Hull–White Not Fully Specified

### Location
IRS references in valuation section.

### What is wrong
The float leg valuation method under Hull–White is not specified.

Possible implementations:
- Par approximation (N(1 − P(t,T)))
- Model-consistent projection using forward rates
- Multi-curve framework

### Why this matters
Different float-leg treatments give materially different exposure profiles and CVA.

The implementation must clearly state which assumption is used.

---

## 6️⃣ Discounting Consistency in CVA Formula Not Explicit

### Location
CVA formula:
```
CVA ≈ (1 − R) Σ DF(0,t_i) × EE(t_i) × ΔPD_i
```

### What is underspecified
Whether EE(t_i) is:
- Undiscounted exposure at t_i
OR
- Already discounted exposure

### Why this matters
If EE is already discounted to time 0, multiplying again by DF double-discounts. The documentation must explicitly state that EE(t_i) is **undiscounted**.

---

## 7️⃣ Curve Extrapolation Policy Not Defined

### Location
Discount curve interpolation section.

### What is missing
What happens beyond last pillar?

Options:
- Flat zero rate
- Flat forward rate
- Flat discount factor
- Forbid extrapolation

### Why this matters
Monte Carlo valuation may require discount factors beyond the last curve pillar.

---

# Summary

The core mathematical structure of the document is sound (HW dynamics, exposure definition, XVA integration framework).

However, the following must be clarified or corrected for mathematical correctness and implementation robustness:

1. Marginal PD equality (incorrect).
2. Hazard interpolation specification (missing).
3. Survival interpolation monotonicity (missing).
4. Collateral sign conventions (ambiguous).
5. IRS float-leg valuation under HW (underspecified).
6. Discounting consistency in CVA (not explicit).
7. Curve extrapolation policy (missing).

---

End of review.
