# Task 2: Alpha Discovery & Research

**Multi-Alpha Research Lab**

---

**AI use disclosure.** I used AI tools to help refine this report and the code. My
purpose was to learn from the suggestions and strengthen my understanding of the
methods and implementation, not to copy others' work.

---

## Executive summary

The instrument in this dataset **mean-reverts at short horizons**. The trend flags in
the supplied library are contrarian indicators, not trend indicators: the naive
baseline that buys when PB01 is on loses 3.0% a year in an asset that rose 36%.

Six candidate strategies were built, each testing a different hypothesis. Applying
a selection rule written down before the holdout was opened, three were selected:
**alpha_02** (oscillator reversal), **alpha_03** (trend-state fade) and **alpha_05**
(volume-flow continuation).

Two findings matter more than the headline numbers:

1. **The six strategies are not six alphas.** Their return space has a participation
   ratio of 1.98, so they behave like about two independent streams. One strategy
   (alpha_04) is 95% explained by the others.

2. **The best strategy in development was the worst out of sample.** alpha_05 scored
   Sharpe 1.02 on development and **−1.45** on the holdout. It passed every
   significance test we ran. We report this rather than quietly dropping it, because
   it is the most useful result in the study: it shows what our statistical battery
   could not catch, and it is the direct motivation for the diversification argument
   in Task 3.

---

## 1. Research setting

**Data.** 986 tradeable candles, 2018-01-02 to 2021-11-01, after the cleaning
described in the Task 1 note.

**Split.** Development 2018-01-02 → 2020-12-31 (770 candles). Holdout 2021-01-01 →
2021-11-01 (216 candles). Every sign, parameter and selection decision was made on
development data. The holdout was scored once, after the strategy set was frozen.
`ResearchContext.fit_strategies()` refuses to fit on any other split.

**Inputs.** The twenty anonymised signals, and nothing else. Price enters only as
the supervised-learning label and for execution and accounting, and
`BaseStrategy.generate_signal()` raises if it is ever handed a frame containing a
price column.

**Costs.** 0.05% per side on every trade, as mandated.

### The observation everything else follows from

Before designing anything, we measured the forward return following each signal on
the development window. The pattern was consistent and, at first, surprising:

| Signal | Meaning | Forward return when ON | Reading |
|---|---|---|---|
| PB01 | short-term trend up | **negative** in all four years | contrarian |
| BB03 | overbought | **negative**, strongest effect in the library | reversal |
| PB07 | distance from trend reference | **negative** IC in all four years | fade the stretch |
| PB08 | trend strength | negative IC in all four years | fade the stretch |
| VB02 | rising cumulative volume flow | **positive** at 1 day, negative at 10 | flow continues briefly |

The trend family is inverted. That is not a data error; it is the signature of an
instrument where short-horizon liquidity provision is paid. Realised volatility is
16.5% annualised with no daily move beyond ±5%, which is consistent with a liquid
index-like product rather than a single volatile name.

This gave us one primary hypothesis class (reversion, expressed four different ways),
one deliberate counter-hypothesis (volume-flow continuation), and one conditioning
hypothesis (volatility state). We wanted the set to contain something that could
disagree with the rest, otherwise the orthogonality analysis has nothing to find.

---

## 2. The six candidate strategies

Each is one file in `strategies/`, one hypothesis, and a declared horizon. **No
direction is hard-coded anywhere.** Each strategy estimates the sign of its own
drivers from the development window inside `fit()`; a driver whose sign cannot be
distinguished from noise gets weight zero rather than a small weight.

### alpha_01: Trend-Stretch Reversion (PB07, PB08; 10 days)
Price does not travel away from its own trend reference indefinitely. The marginal
buyer at an extended level is paying for a move that has already happened. Position
is proportional to the standardised stretch, so a large gap produces a large trade.

### alpha_02: Oscillator Reversal (BB03, BB04; 5 days)
A bounded oscillator at an extreme is a statement about positioning, not value. When
overbought fires, the buyers who were going to buy have bought. Unlike alpha_01 this
is an *event* strategy: flat 86% of days, taking a decaying position for five days
when a flag fires. Same contrarian family, completely different timing and turnover.

### alpha_03: Trend-State Fade (PB01–PB05; 10 days)
Measures *agreement* among trend definitions rather than distance. A market can be
badly stretched with one flag on, or barely stretched with five on; those are
different statements. Persistent flags mean persistent positions.

### alpha_04: Band Position Reversion (BB06, BB01, BB02; 10 days)
Extension measured against a *volatility* envelope rather than a trend reference, so
the same 1% move reads large in a calm market and small in a violent one. The two
measures disagree most when volatility is changing.

### alpha_05: Volume-Flow Continuation (VB01–VB05; 1 day)
**The one strategy that is not contrarian, and it is here on purpose.** A move on
heavy, rising volume is more likely an institution working an order over several
days than a liquidity air pocket. If this holds it should earn when the reversion
book struggles. Whether it does is for the orthogonality analysis to decide.

### alpha_06: Volatility-Regime Conditioned Reversion (BB07, BB05, BB06; 10 days)
Not about direction at all. Reversion is compensation for providing liquidity, and
liquidity is scarcest when volatility is high. This strategy spends its entire
information budget on *how much*, not *which way*.

### What the fitting actually retained

| Strategy | Drivers retained (sign, overlap-corrected t) | Dropped |
|---|---|---|
| alpha_01 | PB07 −1 (t −1.01) | PB08 |
| alpha_02 | BB03 −1 (t −2.02), BB04 +1 | none |
| alpha_03 | PB01 −1, PB03 −1, PB04 −1 | PB02, PB05 |
| alpha_04 | BB06 −1 (t −0.66) | BB01, BB02 |
| alpha_05 | VB02 +1 (t +2.43), VB01 −1, VB05 −1 | VB03, VB04 |
| alpha_06 | BB06 core −1, volatility interaction +1 (t +4.36) | none |

**A caveat we impose on ourselves.** These t-statistics are computed on overlapping
forward windows (a 10-day return measured daily reuses nine days of every
observation), so they are divided by √h before use. Even corrected they are
*screening* statistics, not significance claims: a 770-day window at a 10-day horizon
contains roughly 77 independent observations. Real significance is established in §3
on the backtest's 770 daily returns.

---

## 3. Is the alpha distinguishable from noise?

Five tests, each repairing a different defect in the naive one.

| Strategy | t | Newey-West t | NW p | FDR p | Bootstrap SR [95% CI] | PSR | DSR | **Permutation p** |
|---|---|---|---|---|---|---|---|---|
| alpha_01 | −0.21 | −0.22 | 0.83 | 0.94 | −0.12 [−1.05, 0.74] | 0.42 | 0.04 | 0.42 |
| alpha_02 | 1.38 | 1.44 | 0.15 | 0.45 | 0.79 [−0.28, 1.79] | 0.93 | 0.40 | **0.008** |
| alpha_03 | 0.93 | 0.85 | 0.39 | 0.79 | 0.53 [−0.68, 1.74] | 0.82 | 0.25 | **0.097** |
| alpha_04 | −0.28 | −0.27 | 0.79 | 0.94 | −0.16 [−1.20, 0.88] | 0.39 | 0.03 | 0.39 |
| alpha_05 | 1.79 | 1.85 | 0.064 | 0.38 | 1.02 [0.12, 2.01] | 0.97 | 0.57 | **0.020** |
| alpha_06 | −0.08 | −0.08 | 0.94 | 0.94 | −0.04 [−1.07, 0.96] | 0.47 | 0.05 | 0.34 |

**How to read this honestly.** Not one strategy clears a conventional 5% bar on the
Newey-West t, and none survives the FDR correction for having tested six candidates.
On 770 daily observations that is the expected outcome for effects of this size: a
Sharpe of 0.8 over three years does not produce a t of 2. We say so rather than
quoting the one test that looks best.

The **permutation test** is the one we weight most heavily, because it asks the
question that actually matters. It cuts the position path into blocks and reorders
them, destroying the *timing* while leaving turnover and position sizes intact. If
performance survives that, it came from exposure, not from timing. Alpha_02, alpha_03
and alpha_05 reject the shuffled-timing null; alpha_01, alpha_04 and alpha_06 do not.

The bootstrap confidence intervals carry the same message: only alpha_05's excludes
zero, and it excludes it barely.

---

## 4. Robustness

| Strategy | Sub-periods positive | Break-even cost | Sharpe @ 2× cost | Noise retention (5% corruption) | Parameter plateau | Decay slope | Turnover |
|---|---|---|---|---|---|---|---|
| alpha_01 | 1/3 | 1.3 bps | −0.30 | n/a | 0.00 | +0.24 | 40.7× |
| alpha_02 | 2/3 | **≥25 bps** | +0.66 | 0.45 | **0.83** | +0.25 | **12.3×** |
| alpha_03 | 2/3 | **≥20 bps** | +0.41 | 0.82 | 0.34 | +0.04 | 34.0× |
| alpha_04 | 1/3 | 1.3 bps | −0.40 | n/a | 0.00 | +0.28 | 48.2× |
| alpha_05 | **3/3** | **≥20 bps** | +0.63 | 0.83 | **0.95** | −0.16 | **98.5×** |
| alpha_06 | 1/3 | 3.8 bps | −0.26 | n/a | −0.42 | +0.27 | 49.0× |

**Cost sensitivity.** alpha_01, alpha_04 and alpha_06 do not survive the cost they
must actually pay. They break even around 1–4 bps per side against a mandated 5.
Their edges are real in the sense of being measurable, and irrelevant in the sense of
being uncollectable. Alpha_02 is the standout: it still works at five times the
mandated cost, because it trades 12× a year rather than 50–100×.

**Parameter sensitivity.** Each strategy declares a grid its own hypothesis permits
(`PARAM_GRID`), and the sweep refits at every point. The plateau score is mean Sharpe
divided by best Sharpe: near 1 means a broad plateau where neighbouring settings all
work; near 0 means a spike. Alpha_02 (0.83) and alpha_05 (0.95) sit on plateaus.
Alpha_06's is negative, meaning most settings lose money and the reported figure is
the lucky corner of its grid.

**Noise robustness.** Flipping 5% of the boolean flags at random: alpha_03 and
alpha_05 retain over 80% of their Sharpe, alpha_02 45%. (Retention is undefined for
strategies whose clean Sharpe is negative, so it is left blank rather than reported
as a misleading number above 1.)

**Where they fail.** For alpha_01, alpha_02, alpha_03 and alpha_06, **100% of the
worst drawdown windows are periods where the strategy was positioned against the
market's direction**. It isn't bad luck and it isn't a bug. It is the reversion
trade working exactly as designed and losing, which is what reversion does in a
sustained one-way move. The failure is systematic, bounded and understood. Alpha_05
fails differently: only 60% of its bad windows involve fighting the trend, consistent
with its losses coming from volume spikes that marked capitulation rather than
accumulation.

---

## 5. Independent alpha: are these actually different?

This is the part the problem statement is most specific about, and correlation is
not the answer to it. Two strategies can be mildly correlated pairwise and still add
nothing once a third is in the book.

### Effective dimensionality

Stacking the six return streams as columns of a 770×6 matrix and studying its span:

- **6 numerically independent directions** (trivially, since no two columns are identical)
- **4 components explain 95% of the variance**
- **Participation ratio: 1.98**
- Condition number: 12.4

The participation ratio is the number that matters. It is a cutoff-free "effective
number of independent streams", and it says these six strategies behave like **about
two**. That is the honest headline, and it was expected: four of the six express the
same economic idea through different signals.

### Pivoted QR: ranking by incremental contribution

At each step, column pivoting selects the strategy carrying the most information the
already-selected set does not have.

| Order | Strategy | New information | Already spanned |
|---|---|---|---|
| 1 | alpha_03 | 100% | 0% |
| 2 | alpha_05 | **96.1%** | 3.9% |
| 3 | alpha_01 | 44.6% | 55.4% |
| 4 | alpha_06 | 22.6% | 77.4% |
| 5 | alpha_02 | 78.6% | 21.4% |
| 6 | **alpha_04** | **4.8%** | **95.2%** |

alpha_05 enters second and is 96% new, so the counter-hypothesis paid off. Alpha_04
enters last carrying essentially nothing: 95% of its variation is already available
from the strategies above it. Its pairwise correlation with alpha_06 is **0.965**;
they are near-duplicates measuring band extension two slightly different ways.

### Residual contribution after hedging out everything else

Regressing each strategy on all five others and scoring what is left:

| Strategy | Annualised alpha | Residual Sharpe | R² vs others |
|---|---|---|---|
| alpha_02 | +4.2% | **+1.00** | 0.26 |
| alpha_05 | +9.5% | **+0.88** | 0.25 |
| alpha_03 | +5.6% | **+0.63** | 0.58 |
| alpha_06 | +1.6% | +0.56 | 0.94 |
| alpha_01 | −1.3% | −0.26 | 0.79 |
| alpha_04 | −1.3% | −0.57 | 0.95 |

*(This is the strategy hedged with the others, i.e. the intercept plus the residual, not
the bare OLS residual, which has mean zero by construction and would report 0.00 for
everything.)*

### Stability

Re-running the decomposition on rolling windows: alpha_03 is picked first in **100%**
of windows, and the complete pivot ordering holds in 52% of them. The first direction
is structural; the ordering further down is not, which is itself a reason not to lean
on fine distinctions between the lower-ranked strategies.

### The correlation matrix, for context

alpha_05 is negatively correlated with **every** other strategy (−0.20 to −0.45).
That is what a genuine diversifier looks like, and it came from deliberately building
one hypothesis that contradicted the others rather than from a parameter search.

### The caveat we carry forward

A mathematically new direction is not proof of economic independence. Alpha_02 and
alpha_03 are linearly separable but both express short-horizon reversion; in a regime
where reversion stops working they will fail together regardless of what the QR says.
We read the linear algebra alongside the hypotheses, never instead of them.

---

## 6. Selection

The rule was written down before the holdout was opened, and is structured as gates
then a rank because the problem statement asks for the criteria to be weighed
*jointly*.

**Gates** are the questions with a right answer. Failing any one disqualifies.
- **G1 Timing:** permutation p ≤ 0.30. Did the performance come from *when* it traded?
- **G2 Cost:** break-even cost ≥ 5 bps per side. Does it survive what it must pay?
- **G3 Incremental:** residual Sharpe > 0. Does it add anything the rest does not have?

**Rank** covers the matters of degree: mean z-score across dev Sharpe, sub-period
consistency, decay slope, noise retention and turnover efficiency. Keep everything at
or above the candidate-set average, with a floor of three strategies.

| Strategy | Dev Sharpe | Perm p | Break-even | Residual SR | Gates | Composite | Selected |
|---|---|---|---|---|---|---|---|
| alpha_01 | −0.12 | 0.42 | 1.3 bps | −0.26 | ✗✗✗ | −0.33 | no |
| **alpha_02** | 0.79 | 0.008 | 25 bps | +1.00 | ✓✓✓ | **+0.48** | **yes** |
| **alpha_03** | 0.53 | 0.097 | 20 bps | +0.63 | ✓✓✓ | **+0.07** | **yes** |
| alpha_04 | −0.16 | 0.39 | 1.3 bps | −0.57 | ✗✗✗ | −0.31 | no |
| **alpha_05** | 1.02 | 0.020 | 20 bps | +0.88 | ✓✓✓ | **+0.19** | **yes** |
| alpha_06 | −0.04 | 0.34 | 3.8 bps | +0.56 | ✗✗ | −0.28 | no |

**Selected set: alpha_02, alpha_03, alpha_05.**

The three exclusions are clean. Alpha_01, alpha_04 and alpha_06 each fail the timing
test *and* the cost test: their measured edges do not survive the trade. Alpha_04
additionally contributes almost nothing incremental, and alpha_06, which correlates
0.965 with alpha_04, would have added a second copy of the same idea.

---

## 7. The holdout, opened once

| Strategy | Dev Sharpe | **Holdout Sharpe** | Holdout return | Max DD | Selected |
|---|---|---|---|---|---|
| alpha_01 | −0.12 | +0.21 | +1.1% | −5.4% | no |
| **alpha_02** | 0.79 | **+0.64** | +1.1% | −1.2% | **yes** |
| **alpha_03** | 0.53 | **+2.00** | +14.1% | −6.8% | **yes** |
| alpha_04 | −0.16 | +1.46 | +7.1% | −3.7% | no |
| **alpha_05** | 1.02 | **−1.45** | −8.3% | −11.2% | **yes** |
| alpha_06 | −0.04 | +0.76 | +3.8% | −4.7% | no |
| *buy and hold* | *0.62* | *+0.28* | *+2.0%* | *−5.8%* | |

### What went right

Two of the three selected strategies worked. Alpha_02 held almost exactly its
development Sharpe (0.79 → 0.64), the most stable result in the study, and the one
we would have most confidence deploying. Alpha_03 improved sharply (0.53 → 2.00).
Both beat buy-and-hold on a fraction of its volatility.

### What went wrong, and what we learn from it

**alpha_05 collapsed: Sharpe 1.02 → −1.45.** It was the best strategy on development
by a clear margin. It had the lowest permutation p-value. It had the widest parameter
plateau (0.95). It was positive in all three development sub-periods. It survived 20
bps of cost and 5% signal corruption. It passed every gate we set.

Reviewing what *was* visible before the holdout, two things were:

1. **It was the only strategy with a negative decay slope (−0.16).** Its Sharpe fell
   from 1.36 in the first half of the development window to 0.98 in the second while
   every other strategy improved. The effect was already fading while we were
   measuring it, and the trend was not significant (t −0.91) so our rank penalised it
   only mildly.
2. **Turnover of 98.5× a year**, against 12× for alpha_02. That is 4.9%/yr of cost
   drag on a 12.8% gross return. A strategy needing that much trading to express a
   one-day effect is fragile to any deterioration in the effect itself, and that is
   exactly what happened.

Both were in our scorecard. Neither was weighted heavily enough. Alpha_05 was retained
partly because the rule's three-strategy floor pulled it in at a composite of +0.19.

We report this rather than rewriting the rule and presenting a cleaner-looking set.
The value of a pre-declared rule is entirely destroyed the moment it is revised after
seeing the outcome, and a study that never shows a failure has usually hidden one.

### The other side of the ledger

Three excluded strategies did well on the holdout: alpha_04 (+1.46), alpha_06 (+0.76)
and alpha_01 (+0.21). The gates that excluded them were cost gates, and 2021 was a
low-volatility year (8.5% versus 16–18% earlier) in which those strategies' small
per-trade edges happened to survive. We do not treat this as evidence the exclusions
were wrong: a strategy that breaks even at 1.3 bps per side against a mandated 5 is
not viable, and one favourable year does not change that.

---

## 8. Answers to the required research questions

**What behaviour motivates each strategy?** §2: short-horizon overreaction for
alphas 01–04 and 06, order-working for alpha_05.

**Which signals provide the information?** PB07/PB08; BB03/BB04; PB01–PB05; BB06 with
BB01/BB02; VB01–VB05; BB07/BB05 conditioning BB06. Retained drivers in §2.

**What horizon and rule?** Declared on each class as `fit_horizon`: 10, 5, 10, 10, 1
and 10 days. Rules in §2.

**Is the alpha distinguishable from noise?** Partly. No strategy clears a 5%
Newey-West bar or survives FDR correction on 770 observations. Three reject the
shuffled-timing null. §3.

**Does the effect persist?** Mixed and documented per strategy. Alpha_02 is the most
persistent; alpha_05 was already decaying within the development window. §4, §7.

**Sensitivity to costs, turnover and parameters?** Decisive. Three of six do not
survive the mandated cost. Full break-even, plateau and turnover figures in §4.

**Where does it fail, and is failure systematic?** Yes, systematically: 100% of the
worst windows for the reversion family are periods spent positioned against the
market. §4.

**Are these different sources of alpha?** Partly, and less than the file count
suggests. Participation ratio 1.98 for six strategies; alpha_04 is 95% redundant;
alpha_05 is the only genuine diversifier. §5.

---

## 9. Carried into Task 3

1. The selected set is **alpha_02, alpha_03, alpha_05**, frozen. That includes alpha_05
   despite its holdout failure, because it was selected before the holdout was opened.
2. The set is effectively **two independent directions**, not three. Allocation must
   assume less diversification than the count implies.
3. **alpha_05 is the stress test.** A strategy in the book lost 8.3% on the holdout.
   Whether the portfolio survives that is the question Task 3 has to answer, and it
   is a far more informative question than it would have been had we quietly dropped
   the strategy that failed.

---

### Figures

`results/figures/`: equity curves, drawdowns, correlation heatmap, QR contribution
decomposition, variance spectrum, cost-sensitivity curves, development-vs-holdout
Sharpe.

### Reproducing

```bash
python main.py --task 2
```

Tables in `results/task2_*.csv`; full console output in `results/run_log.txt`.
