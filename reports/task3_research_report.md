# Task 3: Alpha Portfolio Construction & Dynamic Allocation

Multi-Alpha Research Lab

Every number below comes from `results/run_log.txt` or the `results/task3_*.csv`
tables written by `python main.py`.

---

**AI use disclosure.** I used AI tools to help refine this report and the code. My
purpose was to learn from the suggestions and strengthen my understanding of the
methods and implementation, not to copy others' work.

---

## Summary

We allocate across four strategies: the three frozen at the end of Task 2 (alpha_02,
alpha_03, alpha_05) and one new strategy, alpha_07, written after the Task 2 holdout had
already been opened. Section 1 explains why we added it anyway and why its holdout
numbers should be discounted.

Holdout results (1 Jan to 1 Nov 2021, same data, execution rules and 0.05% per side cost
for every method):

| Method | Ann. return | Vol | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|
| 1. Best individual (alpha_07, picked on dev) | +10.80% | 5.8% | 1.80 | −2.5% | 27× |
| 2. Equal weight | +4.93% | 3.0% | 1.59 | −2.4% | 35× |
| 3. Risk-based (risk parity) | +2.89% | 2.4% | 1.21 | −1.9% | 29× |
| 4. Optimisation-based (shrunk max Sharpe) | +0.83% | 2.1% | 0.41 | −1.9% | 30× |
| 5. Dynamic (skill-gated tilt on risk parity) | +2.89% | 2.4% | 1.21 | −1.9% | 29× |
| *Ref: Task 2 book, equal weight, no alpha_07* | *+2.77%* | *2.9%* | *0.95* | *−2.4%* | *43×* |
| *Ref: buy and hold* | *+2.00%* | *8.5%* | *0.28* | *−5.8%* | — |

The main findings:

1. **The learned component doesn't work, and we don't use it.** Its out-of-sample
   cross-sectional IC is +0.069 against a shuffled-label null whose 95th percentile is
   +0.166 (p = 0.24, 200 permutations). With the gate switched off, its tilt doesn't beat
   random tilts on the development window (p = 0.18) or on the holdout (p = 0.87). The
   skill gate never opened during the holdout, so method 5 held its risk parity base and
   matched method 3 exactly.
2. **Estimation hurt out of sample, again.** The fewer inputs a method estimates, the
   better it did on the holdout: equal weight > risk parity > max Sharpe. Same order as
   on 18 Sep, before alpha_07 existed.
3. **alpha_07 is where the improvement came from.** Equal weight with alpha_07 beat the
   frozen Task 2 book by 2.1%/yr on the holdout (Newey-West p = 0.085) and 2.0%/yr over
   the full history. Its development-window evidence is good on its own terms, but we
   wrote it after seeing Task 2's holdout, so the clean out-of-sample number in this
   project is still the Task 2 book's +2.77%.

**Recommendation: equal weight across the four strategies, with the learned tilt left
switched off.** Reasons are in §8.

---

## 1. What is being allocated

| Strategy | Idea | Signals | Came from |
|---|---|---|---|
| alpha_02 | fade oscillator extremes for a few days | BB03, BB04 | Task 2 selection |
| alpha_03 | fade agreement among trend flags | PB01–PB05 | Task 2 selection |
| alpha_05 | volume-backed moves continue for a day | VB01–VB05 | Task 2 selection |
| alpha_07 | long by default, leans against overbought readings and trend-flag crowding | BB03, BB04, PB01, PB03 | added at Task 3 |

### Why alpha_07 exists

After Task 2 we looked at the book's positions and found a problem we had missed. Every
Task 2 strategy trades reversion around zero, and because most reversion signals say
"trend is up, so fade it", the book leaned slightly short. The average net position
across the six candidates on the development window was −0.06, while the instrument went
up 9.7% a year. That drift was just left on the table. Over the full history the
equal-weight Task 2 book made 7.3%/yr against buy and hold's 8.2%, at a far better Sharpe
but a lower return.

alpha_07 separates two decisions that the other strategies had merged: what to hold when
there is no view (a long anchor, whose size is fitted on dev and could have come out as
zero), and how far to lean away from it (contrarian overlays). The fit chose anchor 1.0,
BB03 faded over 5 days, trend consensus faded over 8 days, gain 1.25. BB04 didn't clear
the sign screen on dev and got weight zero.

With those settings the overlays are strong enough to flip the position, not just trim
it. On dev alpha_07 was fully long on 41% of days and short on 34% (mostly when both trend
flags had been on for a while), with an average net position of +0.35. So a large part of
its return is market exposure, and the factor model in §2 measures how large.

### Why it isn't in the Task 2 run

The Task 2 selection rule was frozen before the holdout was opened, and that was the
whole point of it. If alpha_07 goes into the Task 2 candidate list, the rule picks
{alpha_07, alpha_02, alpha_05} and drops alpha_03. That would quietly rewrite a result
already submitted, and alpha_03 happens to be the strategy with the best holdout (Sharpe
2.00). So `strategies/ALPHA_REGISTRY` still holds only alpha_01–06, the Task 2 outputs
are reproduced exactly (we checked all 48 tables against the 18 Sep run), and alpha_07
comes in through a separate `TASK3_ADDITIONS` registry.

### Its evidence, on the same tests Task 2 used

Everything below is on the development window, with the same code as Task 2.

| Test | alpha_07 | Task 2 gate |
|---|---|---|
| Dev Sharpe | 1.07 (buy and hold 0.62) | |
| Permutation p (shuffled timing) | 0.048 | ≤ 0.30, passes |
| Break-even cost | ≥ 25 bps per side | ≥ 5 bps, passes |
| Residual Sharpe after hedging out alpha_02/03/05 | +0.55 | > 0, passes |
| Newey-West t | 1.85 (p = 0.065) | |
| Bootstrap Sharpe 95% CI | [−0.10, 2.23] | |
| Deflated Sharpe probability, best of 7 ideas (default trial spread) | 0.56 | |
| Deflated Sharpe probability, best of the 180-setting grid (observed spread) | 0.79 (lucky best-of-grid Sharpe would be 0.61) | |
| Parameter grid: share of 180 settings with Sharpe > 0 | 100% (plateau score 0.74) | |
| Sharpe at 2× cost | 0.98 | |
| Sub-periods positive | 2 of 3 (2019 lost 5.5%, buy and hold lost 5.8%) | |
| Turnover | 25× a year | |

It would have passed all three Task 2 gates. Like everything else in this project it
doesn't clear a 5% Newey-West bar on 770 days. On the holdout it made +10.8% at Sharpe
1.80.

We don't want to lean on that holdout figure. The parameters were fitted on dev only, but
the *idea* came from looking at results that included 2021. What we can defend is that
the idea is visible from dev alone (buy and hold returned 9.7%/yr on dev), and that the
anchor grid included zero, so the fit was free to reject the idea. It didn't. The real
test for alpha_07 is the organisers' unseen data.

---

## 2. Factor model: how much is alpha and how much is exposure

    r_i(t) = alpha_i + beta_i' F(t) + eps_i(t)

With one instrument there's no cross-section to build size, value or momentum factors
from, so the factors come from the instrument's own time series:

| Factor | Definition | Why it's there |
|---|---|---|
| MKT | buy-and-hold return, open to open | anything net long picks this up, and alpha_07 is net long |
| VOL | change in 21-day realised vol, scaled to return units | reversion strategies get hurt when calm turns violent; without this that shows up as alpha |
| REV | minus yesterday's sign × today's return | the return every reversion strategy shares; without it each would claim the same alpha |

The factors use price, which is fine here because they are only used for attribution and
never reach a trading decision. Standard errors are Newey-West.

**Development window** (`task3_factor_model_dev.csv`)

| Strategy | Alpha /yr | alpha t | β MKT (t) | β VOL (t) | β REV (t) | R² |
|---|---|---|---|---|---|---|
| alpha_02 | +5.5% | 2.20 | −0.11 (−4.8) | +0.64 (2.8) | +0.00 (0.1) | 0.18 |
| alpha_05 | +5.4% | 0.85 | +0.20 (3.4) | +0.86 (1.6) | −0.22 (−4.1) | 0.18 |
| alpha_03 | +10.6% | 1.29 | +0.05 (0.6) | +2.01 (2.7) | +0.10 (1.8) | 0.04 |
| alpha_07 | +11.5% | 1.66 | +0.48 (6.6) | +1.51 (2.6) | +0.01 (0.1) | 0.33 |

**Holdout** (`task3_factor_model_holdout.csv`, a diagnostic only, not used for anything)

| Strategy | Alpha /yr | alpha t | β MKT (t) | R² |
|---|---|---|---|---|
| alpha_02 | +1.2% | 1.15 | −0.09 (−3.0) | 0.23 |
| alpha_05 | −8.7% | −1.70 | +0.04 (0.4) | 0.08 |
| alpha_03 | +14.2% | 2.64 | −0.45 (−4.5) | 0.33 |
| alpha_07 | +9.4% | 1.88 | +0.23 (2.1) | 0.16 |

What we take from this:

- alpha_07 carries about half a unit of market beta on dev, but its overlay still earns
  an intercept of about 11%/yr on top of that, and on the holdout it did the same (9.4%,
  t = 1.88). Its outperformance was not just 2021 being an up year. Buy and hold only made
  2% in 2021.
- alpha_02 is the only strategy with a clearly significant dev alpha, and it earns it
  while slightly short the market.
- alpha_05 is the only strategy loading negatively on REV (t = −4.1). That is the
  factor-model version of Task 2's finding that it is the one real diversifier, and it is
  why the book held up when alpha_05 lost 8.7%/yr of alpha on the holdout.
- On dev, every strategy loads positively on VOL. The book as a whole is long "volatility
  change", and correlations alone don't show that. It is the biggest shared risk in the
  portfolio.

---

## 3. Costing the book as one account

Scoring an allocation as a weighted sum of each strategy's net returns double-charges:
if alpha_02 goes from +1 to 0 on the day alpha_03 goes from 0 to +1, one account holding
both trades nothing. So every allocation here is rebuilt from positions,

    combined_position(t) = sum_i w_i(t) * position_i(t)

and that one position goes through the same `ExecutionEngine`, `Portfolio` and
`PerformanceAnalyzer` as a single strategy (`portfolio_backtest.py`). Rebalancing cost
needs no separate charge because a weight change shows up as a change in the combined
position.

| Equal weight, full history | Per year |
|---|---|
| Turnover if the strategies were separate accounts | 43.8× |
| Turnover as one account | 33.1× |
| Cost as separate accounts | 2.19% |
| Cost as one account | 1.65% |
| Saving | 0.53% |

About half a percent a year is a tenth of the equal-weight holdout return, so netting
isn't a rounding detail.

---

## 4. Static allocation

Weights are fitted on the development window and then held fixed.

| Strategy | Equal weight | Inverse vol | Risk parity | Max Sharpe | Mean-variance |
|---|---|---|---|---|---|
| alpha_02 | 25.0% | 48.1% | 47.8% | 60.0% | 0.0% |
| alpha_05 | 25.0% | 18.8% | 22.0% | 24.1% | 50.3% |
| alpha_03 | 25.0% | 17.1% | 16.3% | 5.8% | 0.0% |
| alpha_07 | 25.0% | 16.0% | 13.9% | 10.1% | 49.7% |

Weights hide how concentrated a book really is when the strategies' vols differ by three
times (alpha_02 runs at 5% vol, alpha_07 at 15%). The share of portfolio variance each
strategy contributes, computed on the same shrunk covariance the optimiser uses
(`task3_risk_contributions_dev.csv`):

| Strategy | Equal weight | Risk parity | Max Sharpe | Mean-variance |
|---|---|---|---|---|
| alpha_02 | 7% | 25% | 40% | 0% |
| alpha_05 | 20% | 25% | 36% | 43% |
| alpha_03 | 33% | 26% | 7% | 0% |
| alpha_07 | 39% | 25% | 17% | 57% |

Equal weight looks diversified by weight, but 73% of its risk sits in alpha_03 and alpha_07,
the two strategies built on the trend flags. We come back to this in §8.

Assumptions.
- Expected returns are shrunk halfway to their cross-sectional mean. Sample means over
  770 days have standard errors about as large as the means themselves, and without
  shrinkage the optimiser piles into whoever got lucky. alpha_07's sample mean of 15.8%
  becomes 12.9%, and alpha_02's 3.9% becomes 6.9%.
- The covariance matrix is shrunk halfway towards constant correlation. With 4 strategies
  and 770 days there are 55 observations per parameter and the condition number is 13.9,
  so the covariance is usable, but the optimiser will still exploit any noise it finds.
- Long-only, fully invested, 60% cap per strategy. Shorting a whole strategy is too
  strong a claim for three years of data.

Mean-variance split the book 50/50 between the two strategies with the largest sample
means, alpha_05 and alpha_07, and alpha_05 then lost 8.3% on the holdout. Max Sharpe,
starting from the same shrunk inputs, put 60% in alpha_02. Two objectives on the same
inputs give opposite books, which tells you how much of an optimiser's answer comes from
the objective rather than the data.

---

## 5. The learned component

### What it predicts

The Tech Doc rules out a generic next-price model, and an allocator doesn't need one. The
label is strategy effectiveness over the next rebalance block:

    y_i(k) = mean net return of strategy i over block k+1 / its trailing 3-block vol

Features, all known at the end of block k: each strategy's trailing mean return (1 and 3
blocks), trailing vol, recent/longer vol ratio, hit rate and current drawdown, plus four
market-state numbers from the signal library (average PB, BB and VB flag levels, and
BB07). Price and volume never go in. We use a strategy's own past P&L to size it, and we
treat that as an allocation input rather than a trading signal. That is our reading of
the rules, stated here so it can be checked.

### Capacity

The development window gives 33 usable 21-day blocks. With four strategies that is 132
rows, but the rows are not independent: all four strategies in a block share the same
market. With 10 features that is 3.3 independent blocks per feature. So:

- ridge regression, linear, no interactions, no trees
- 10 slow features, fixed before fitting
- the ridge penalty is chosen by time-ordered CV *inside* each training fold, split on
  block boundaries, so the amount of regularisation never sees the future (it settles on
  100, which is heavy). The first four folds have fewer than 8 training blocks, too few
  to split, and use a fixed penalty of 10.
- features standardised with training-fold statistics only

A gradient-boosted model would post a much better in-sample number on this data, and
that's exactly why we didn't use one.

### Walk-forward validation

Expanding window, refit at every block, 29 test folds (`task3_meta_walkforward.csv`).
Each test fold is one block, so four predictions.

Which statistic counts matters here. The allocator z-scores the predictions across
strategies inside each block before it tilts, so only the ranking within a block ever
reaches the weights. The allocation-relevant statistic is therefore the cross-sectional
IC (rank correlation across the four strategies in a block, averaged over blocks). R² is
pooled over all out-of-sample predictions because R² on four points is meaningless.

The three columns have to come from the same folds, so this table uses the 25 folds that
have an inner-CV validation score:

| 25 folds | Train | Validation (inner CV) | Test |
|---|---|---|---|
| R² | +0.162 | −0.032 | +0.030 (pooled) |
| Cross-sectional IC | +0.134 | +0.175 | +0.008 |

- train → test R² gap: +0.132
- train → test IC gap: +0.127
- over all 29 folds: mean test cross-sectional IC +0.069, positive in 15 of 29

The model explains about a sixth of the variance it's trained on and close to none of
what it hasn't seen, and on these 25 folds its ranking skill is essentially zero. The
validation IC coming out above the train IC isn't a good sign; it's what small, noisy
folds look like.

### Null baseline

The whole walk-forward is re-run 200 times with labels shuffled *within* each block. That
keeps each month's set of outcomes and destroys only the link between a strategy's
features and its own outcome, which is the one thing the allocator would rely on.

| | Value |
|---|---|
| Real model, mean test cross-sectional IC | +0.069 |
| Null mean | −0.013 (sd 0.111) |
| Null 95th percentile | +0.166 |
| Permutation p | 0.239 |

**It does not clear the null.** A model trained on shuffled labels does at least this
well about a quarter of the time, so we report it as indistinguishable from noise.

(An earlier version of this report quoted pooled IC against a null built on per-block
IC, which mixed two different statistics. Both sides now use the same statistic.)

### What would have to be true for the result to be real

The Tech Doc asks for this explicitly.

For the +0.069 to be a real, small skill rather than noise, we'd expect:

- Stable sign across folds. It isn't: 13 of 29 folds are negative (one is exactly
  zero), and they aren't bunched in one period.
- Most of the model's weight on the features that can change a ranking. That isn't
  there either. The largest coefficients are on market-state features (band flags
  −0.047, vol level +0.044). Those take the same value for all four strategies in a
  block, and in a linear model with no interactions they move all four predictions by the
  same amount, which the allocator's within-block z-score removes. The model spends most
  of its capacity on timing ("next month is good for everyone"), and that is why the
  pooled IC (+0.28) is so much higher than the cross-sectional IC (+0.07). Making those
  features matter would need strategy × state interactions, which quadruples the feature
  count on 33 blocks. The capacity argument above rules that out.
- Out-of-sample labels. The strategies' own parameters were fitted on dev, so the
  effectiveness labels the model learns from on dev are partly in-sample. If the model
  had shown skill, part of it could have been fit to that. Since it shows none, this
  doesn't change the verdict, but it would matter for anyone trying to rescue the model.

The pooled IC of +0.28 hints that the features might say something about when the whole
book does well. That would be a question about gross exposure, not allocation, and we
haven't tested it against a null, so we make no claim about it.

---

## 6. Dynamic allocation

### How it turns scores into weights

The model's predictions pass through several rules before any weight changes, and those
rules matter more to the result than the model does (`dynamic_allocator.py`):

1. Tilt, don't rebuild. Scores are z-scored within the block, turned into a
   multiplicative tilt on the risk parity weights (strength 0.5) and blended 50/50 with
   the base.
2. No-trade band. Weights only move if some strategy's target has drifted more than 5
   points.
3. Caps. No strategy above 50%.
4. Skill gate. The tilt is multiplied by a confidence number that is zero unless the
   model has at least 6 closed blocks of out-of-sample IC with a positive mean and a
   t-stat of at least 1.0. Only blocks that have already closed count.
5. Timing. A block's last daily strategy return runs from one open to the next, so it
   is only known at that next open. New weights therefore trade one open later. An
   earlier version traded at that same open, which is a one-day look-ahead; we found it
   in review and removed it. It had been flattering the model on dev (see below).

The gate exists because of §5. A component that can't show skill shouldn't move money, and
we wanted that decided by a rule written in advance rather than by us after seeing the
result.

### What it did

Run once, continuously, from 2018 to 2021, refitting at every block.

- The gate was open at 7 of 43 rebalances, all between January 2019 and September 2020.
  It never opened in the holdout.
- It traded at 11 rebalances (including the trades back to base when the gate shut), for
  a total weight turnover of 2.0×.
- Average weights: alpha_02 47%, alpha_05 22%, alpha_03 17%, alpha_07 14%.

So in the holdout the dynamic method was risk parity, which is why rows 3 and 5 of the
summary table are identical.

### The allocation-level null

That identity also means the comparison table can't tell us anything about the model.
To test the model itself, we switched the gate off, let the model tilt freely, and
compared it with 200 runs of the same allocator fed random scores instead of
predictions. Random scores go through the same z-score, tilt, blend, cap and no-trade
band, so their turnover is comparable. This is the "randomised allocation of equivalent
turnover" null the Tech Doc mentions (`task3_tilt_null.csv`,
`fig_task3_tilt_null.png`).

| Window | Model Sharpe − base Sharpe | Random: mean | Random: 95th pct | p |
|---|---|---|---|---|
| Dev, after warm-up (Aug 2018 – Dec 2020) | +0.08 | −0.01 | +0.16 | 0.18 |
| Holdout (2021) | −0.29 | −0.02 | +0.31 | 0.87 |

The random tilts turned over 7.5× in total against the model's 5.9×, so they paid a
little more in costs. That biases the test slightly in the model's favour, and the model
still doesn't clear it on either window. On the holdout its tilt was worse than most
random ones. With the one-day look-ahead still in, the dev result had been +0.12 at
p = 0.085, which looked borderline; that was the leak, not the model.

---

## 7. The required comparison

Same data, same open-to-open execution, same 0.05% per side, same windows, every method
costed from netted positions through the same code. "Best individual" is picked on dev
Sharpe (alpha_07 at 1.07 against alpha_05 at 1.02) and carried forward unchanged.

### Holdout: 2021-01-01 to 2021-11-01

| Method | Total | Ann. return | Vol | Sharpe | Sortino | Calmar | Max DD | Cost drag |
|---|---|---|---|---|---|---|---|---|
| 1. Best individual | +9.2% | +10.80% | 5.8% | 1.80 | 2.95 | 4.36 | −2.5% | 1.34% |
| 2. Equal weight | +4.2% | +4.93% | 3.0% | 1.59 | 2.62 | 2.09 | −2.4% | 1.76% |
| 3. Risk parity | +2.5% | +2.89% | 2.4% | 1.21 | 1.98 | 1.55 | −1.9% | 1.45% |
| 4. Max Sharpe | +0.7% | +0.83% | 2.1% | 0.41 | 0.66 | 0.44 | −1.9% | 1.48% |
| 5. Dynamic | +2.5% | +2.89% | 2.4% | 1.21 | 1.98 | 1.55 | −1.9% | 1.45% |
| *Task 2 book, EW* | *+2.4%* | *+2.77%* | *2.9%* | *0.95* | *1.52* | *1.15* | *−2.4%* | *2.13%* |
| *Buy and hold* | *+1.7%* | *+2.00%* | *8.5%* | *0.28* | *0.40* | *0.34* | *−5.8%* | — |

### Full history: 2018-01-02 to 2021-11-01

Static weights were fitted on the first 770 of these 986 days, so this table is mostly
in-sample for them.

| Method | Ann. return | Vol | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|
| 1. Best individual | +14.66% | 13.3% | 1.09 | −14.8% | 25× |
| 2. Equal weight | +9.40% | 6.7% | 1.37 | −7.75% | 33× |
| 3. Risk parity | +7.38% | 5.1% | 1.42 | −6.3% | 28× |
| 4. Max Sharpe | +6.38% | 4.4% | 1.44 | −4.9% | 29× |
| 5. Dynamic | +7.28% | 5.2% | 1.38 | −7.0% | 29× |
| *Task 2 book, EW* | *+7.34%* | *5.5%* | *1.31* | *−8.0%* | *39×* |
| *Buy and hold* | *+8.24%* | *16.1%* | *0.57* | *−18.9%* | — |

The development-window table is in `task3_comparison_dev.csv` (in-sample for every
static method by construction).

### Are the differences real?

Newey-West t on the daily return difference (`task3_significance.csv`):

| Comparison | Holdout excess | t | p | Full-history excess | t | p |
|---|---|---|---|---|---|---|
| Dynamic vs best individual | −7.5%/yr | −1.80 | 0.072 | −7.4%/yr | −1.50 | 0.13 |
| Dynamic vs equal weight | −2.0%/yr | −2.15 | 0.031 | −2.1%/yr | −1.90 | 0.058 |
| Dynamic vs risk parity | 0.0 | 0.00 | 1.00 | −0.1%/yr | −0.33 | 0.74 |
| Dynamic vs max Sharpe | +2.0%/yr | +2.52 | 0.012 | +0.9%/yr | +0.93 | 0.35 |
| Equal weight vs best individual | −5.6%/yr | −1.53 | 0.13 | −5.4%/yr | −1.30 | 0.19 |
| Equal weight vs risk parity | +2.0%/yr | +2.15 | 0.031 | +2.0%/yr | +1.74 | 0.082 |
| Equal weight vs max Sharpe | +4.0%/yr | +2.43 | 0.015 | +2.9%/yr | +1.53 | 0.13 |
| Equal weight vs Task 2 book EW | +2.1%/yr | +1.72 | 0.085 | +2.0%/yr | +1.43 | 0.15 |

On the holdout, equal weight earned significantly more than risk parity (p = 0.031) and
max Sharpe (p = 0.015). Over the full history the same gaps shrink to p = 0.08 and 0.13,
and they are gaps in return, not in Sharpe: equal weight simply holds more of the
higher-vol strategies. The dynamic method can't be told apart from risk parity, which is
what a closed gate should produce. No allocation beats holding alpha_07 alone with any
confidence, and that comparison carries alpha_07's hindsight problem anyway.

### Stability: worst third of each window

The organisers say they may re-score on sub-periods, so we did it first
(`task3_subperiods_*.csv`):

| Method | Worst-third Sharpe, holdout | Worst-third Sharpe, full |
|---|---|---|
| 1. Best individual | 1.27 | 0.41 |
| 2. Equal weight | 0.72 | 0.72 |
| 3. Risk parity | 0.79 | 0.85 |
| 4. Max Sharpe | −0.28 | 1.09 |
| 5. Dynamic | 0.79 | 0.75 |

Over the full history alpha_07 alone has the weakest worst third (Jan 2018 to Apr 2019,
Sharpe 0.41), and all four allocations are steadier than it. Max Sharpe is the steadiest over the full history but
the only allocation with a losing third in the holdout.

### Harsher costs

The organisers may also change execution assumptions. Sharpe at 1×, 2× and 4× the
mandated cost (`task3_cost_stress.csv`):

| | Best indiv. | Equal weight | Risk parity | Max Sharpe | Dynamic |
|---|---|---|---|---|---|
| Holdout, 5 bps | 1.80 | 1.59 | 1.21 | 0.41 | 1.21 |
| Holdout, 10 bps | 1.56 | 1.01 | 0.59 | −0.30 | 0.59 |
| Holdout, 20 bps | 1.09 | −0.14 | −0.61 | −1.71 | −0.61 |
| Full, 5 bps | 1.09 | 1.37 | 1.42 | 1.44 | 1.38 |
| Full, 10 bps | 1.00 | 1.12 | 1.15 | 1.11 | 1.11 |
| Full, 20 bps | 0.81 | 0.63 | 0.59 | 0.44 | 0.55 |

This was the most uncomfortable table to produce. The book turns over roughly 30× a
year, mostly because of alpha_05 (98×) and alpha_03 (34×). At twice the mandated cost the
holdout Sharpe of every allocation drops by a third or more, and at four times it is
negative. alpha_07 on its own, at 25×, degrades least. If we had another week, turnover is
where we'd spend it.

---

## 8. Recommendation

**Equal weight across alpha_02, alpha_03, alpha_05 and alpha_07, with the learned tilt
left behind its gate.**

Why equal weight:

- It estimates nothing. Over the full history the four allocations have Sharpe ratios
  between 1.37 and 1.44, and the return differences between them are marginal at best
  (p = 0.08 against risk parity, 0.13 against max Sharpe).
- It has the highest return of the four at a similar Sharpe, which matters because net
  and annualised return are the first things the final evaluation lists.
- On the one window none of them were fitted on, it came first, in the same order as on
  18 Sep before alpha_07 existed. We chose equal weight then, and nothing here gives us
  a reason to change.

What we're not comfortable with:

- Risk concentration. 39% of equal weight's variance comes from alpha_07, the
  strategy with the least clean evidence, and 73% from the two trend-flag strategies
  together. Risk parity caps alpha_07 at a quarter of the risk, cuts the full-history
  drawdown from −7.75% to −6.3%, and costs about 2%/yr of return. For a mandate that cares
  more about drawdown than return, risk parity is the better choice, and it is also what
  the dynamic method falls back to.
- Cost sensitivity. See §7. Both candidates lose most of their edge at 2–4× costs.

Why not the learned tilt: it fails its permuted-label null (p = 0.24) and its
random-tilt null (p = 0.18 on dev, 0.87 on the holdout). The machinery is built, tested and gated, and
it stays gated. The Tech Doc says a smaller validated result should score above a larger
unvalidated one, and here the validated result is "the model doesn't help".

---

## 9. Answers to the Task 3 research questions

| PS asks about | Where |
|---|---|
| Strategy-specific alpha and systematic exposure | §2, dev and holdout factor fits |
| Portfolio volatility, drawdown, concentration | §4 (risk shares), §7 tables |
| Diversification and interactions | §2 (REV and VOL loadings), §3 (netting), Task 2 §5 |
| Turnover and transaction-cost impact | §3, §7 (turnover, cost drag, cost stress) |
| Static vs dynamic allocation | §4, §6, §7 |
| Can time-varying effectiveness be exploited without look-ahead? | §5–6: the model is walk-forward and gated on closed blocks only; it didn't find anything usable |
| Is the learned component real, given the sample size? | §5: capacity, train/validation/test, permuted-label null; §6: random-tilt null. No. |

---

## 10. Reproducing

```bash
pip install -r requirements.txt
python main.py            # all three tasks, about 90 seconds on a laptop
python main.py --quick    # fewer permutations and bootstrap draws, about 40 seconds
```

Task 3 writes `results/task3_*.csv` and `results/figures/fig_task3_*.png`:

| File | What's in it |
|---|---|
| `task3_addition_alpha_07_scorecard.csv` | alpha_07 on the Task 2 tests |
| `task3_factor_model_dev.csv`, `..._holdout.csv` | factor loadings |
| `task3_static_weights.csv`, `task3_risk_contributions_dev.csv` | static allocations, risk shares |
| `task3_meta_walkforward.csv`, `task3_meta_null_baseline.csv` | model discipline evidence |
| `task3_tilt_null.csv` | allocation-level null |
| `task3_allocation_history.csv`, `task3_dynamic_weight_path.csv` | what the dynamic allocator did |
| `task3_comparison_{dev,holdout,full}.csv` | the required comparison |
| `task3_significance.csv`, `task3_subperiods_*.csv`, `task3_cost_stress.csv` | tests on the comparison |
| `fig_task3_*.png` | walk-forward folds, both nulls, weight path, equity curves, drawdowns |
