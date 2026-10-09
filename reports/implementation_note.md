# Task 1: Implementation Note

**Multi-Alpha Research Lab, Data & Backtesting Infrastructure**

---

**AI use disclosure.** I used AI tools to help refine this report and the code. My
purpose was to learn from the suggestions and strengthen my understanding of the
methods and implementation, not to copy others' work.

---

## 1. What this submission is

A research framework in which data handling, strategy logic, execution, portfolio
accounting and performance measurement are separate components that know as little
about each other as possible. The test of that separation is simple: adding a new
strategy means adding one file under `strategies/` and nothing else. No engine code
changes, no special cases.

The whole pipeline runs with one command and writes every table it prints:

```bash
python main.py --task 1
```

## 2. Design decisions, and why

### The decision frame is the only thing a strategy ever sees

The single largest source of accidental look-ahead in backtests is that the
strategy has access to the price series and someone eventually uses it. We removed
the possibility rather than the temptation.

`FeatureEngine.build_decision_frame()` returns two objects. The *decision frame*
contains the twenty signal columns, the date, and an `information_asof` audit
column, and no price or volume at all. The *market frame* contains OHLCV and the
return series and is handed only to the execution engine and the portfolio.
`BaseStrategy.generate_signal()` raises if it is ever passed a frame containing a
price column, so a strategy cannot look at price even by accident.

This directly enforces the rule that a trading decision must be determined by the
signal library alone. Price appears in exactly two roles, both permitted: as the
supervised-learning label when a strategy is fitted, and for execution and
accounting.

### The t−1 cutoff is expressed once, not in six places

Row *t* of the decision frame holds the most recent signal observation *strictly
before* date *t*. That is produced by a single `merge_asof(..., direction="backward",
allow_exact_matches=False)`. There is one line of code to audit rather than a
convention every strategy has to remember.

`Backtester.generate_signals()` re-asserts the property before passing anything to a
strategy, and `FeatureEngine.validate_no_lookahead()` verifies it empirically: it
samples rows, looks up the raw signal file, and confirms that every value in the
decision frame really does come from an earlier date. It raises rather than warns.

### Returns are open-to-open, and this is not cosmetic

The mandated convention is that a decision made after candle *t−1* executes at
candle *t*'s open. A position entered at `open[t]` and held one candle is exited at
`open[t+1]`. The capturable return is therefore `open[t+1]/open[t] − 1`.

Using close-to-close instead (the more common default) would credit every
strategy with the overnight gap between `close[t]` and `open[t+1]`, which it never
had the opportunity to trade. On this dataset that is not a rounding error. The
same convention is used for the research label, so what a strategy is fitted on and
what it is scored on are the same quantity.

### The vectorised engine is checked against a literal loop

The backtest is vectorised for speed, which is only safe because the timing is
baked into the frame rather than into a loop index. To prove the fast path is
faithful, `Backtester.run_reference_loop()` implements the per-candle loop exactly
as written in the Technical Documentation, and `assert_matches_reference()` compares
the two equity curves.

They agree to **6.7 × 10⁻¹⁶**: floating-point noise.

This check earned its place immediately. On its first run it failed by 4.4 × 10⁻⁴
and exposed a real bug: the final candle's position was being zeroed *after* its
cost had been computed, handing every strategy a free exit. The exit is now forced
before costs are charged, in `Backtester.execute_signals()`.

### Costs

0.05% of traded notional per side, charged on `|p_t − p_{t−1}|`, so a full round
trip costs exactly 0.10%. Positions are expressed as a fraction of equity, which is
the natural representation for a single instrument traded from one account, and it
makes the cost in return space simply `c × |Δp|`.

Slippage is modelled as a separate per-side charge on the same notional and is
**zero in the headline results**. The reasoning: the problem statement fixes the
cost assumption and says not to substitute a different one, so adding an invented
slippage figure to the headline would make our numbers incomparable with every
other team's. Slippage instead enters as an explicit robustness dimension, varied
from 0 to 10 bps per side in the cost-sensitivity study.

## 3. What was wrong with the supplied data

The price file is not clean, and the defects are not accidental. Each breaks a
backtest differently, so each is handled explicitly and counted rather than being
absorbed by a generic `dropna()`.

| Defect | Count | How it is handled | Why it matters |
|---|---|---|---|
| Rows out of chronological order | whole file | Sorted before anything else touches the frame | Every return, rolling window and lag assumes row order *is* time order |
| Mixed date formats | 27 rows `DD-MM-YYYY` among 973 `YYYY-MM-DD` | Formats tried in order; unparseable dates raise | A single-format parse silently yields `NaT`; day-first inference silently misreads `2018-01-02` |
| Duplicate rows | 14 dates | Verified byte-identical, then de-duplicated keeping the first; conflicts would be reported loudly | Duplicates double-count a day's return and corrupt every rolling statistic |
| Warm-up gaps | BB06 ×19, BB07 ×13, VB05 ×19 | **Left as NaN**, never back-filled; strategies read NaN as "no opinion" | A backward fill copies a value into days before it existed, which is textbook look-ahead |
| Dates with signals but no price | 14 dates | Excluded from the trading calendar, retained in the information set | There is no open to fill against, so the day is not tradeable: but the signal was still known |

That last row is the subtle one. Those fourteen days are dropped from the *tradeable*
calendar because a fill price does not exist, but their signal values are not thrown
away: the as-of join carries them into the next tradeable day's decision, which is
what a desk would actually have on screen.

After cleaning: **986 tradeable dates**, 2018-01-02 to 2021-11-01, zero OHLC
consistency violations.

## 4. Research-integrity checks

These run every time and raise rather than warn:

| Check | Result |
|---|---|
| Signal information is strictly older than the candle it trades | PASS, 400 rows re-derived from the raw file |
| Decision frame exposes no price or volume column | PASS |
| Vectorised engine reproduces the per-candle reference loop | PASS (max equity difference 6.7e-16) |
| A full round trip costs 0.10% of notional | PASS |
| Trades fill at candle *t*'s open | PASS |
| Development and holdout windows are disjoint | PASS |

Two further disciplines are enforced in code rather than documented and hoped for:

- `ResearchContext.fit_strategies()` **refuses to fit on anything but the development
  window**. Including `'full'`, which contains the holdout.
- Forward-return labels have their final *h* rows blanked, so the label for the last
  development candle cannot be computed from the first holdout price.

## 5. The baseline, and why it is the point

`BaselineStrategy` is the obvious reading of a trend flag: hold the asset while
PB01 is on, flat otherwise. It is deliberately not tuned.

| | Baseline (PB01 long-only) | Buy and hold |
|---|---|---|
| Annualised return | **−3.02%** | +8.24% |
| Sharpe | **−0.19** | +0.57 |
| Max drawdown | −32.1% | −18.9% |
| Annual turnover | 8.7× | — |
| Cost drag | 0.43%/yr | — |

The baseline loses money, after costs, in an instrument that rose 36% over the
period, and takes a worse drawdown than simply owning it. That is the motivating
result for Task 2. The obvious reading of the trend flags is not merely weak here;
it is backwards, and the research that follows starts from that observation.

## 6. Structure

```
main.py                 one entry point; --task 1|2|3, --quick
config.py               paths, the mandated cost, the dev/holdout boundary
data_loader.py          load and schema-check; repairs nothing
data_cleaner.py         all repairs, all diagnostics
feature_engine.py       the decision frame and the information cutoff
strategy.py             BaseStrategy: the replaceability contract
execution_engine.py     fills, costs, slippage, trade log
portfolio.py            positions, equity, P&L, the only place equity is computed
performance.py          every metric, for every strategy and allocation
backtester.py           the loop, plus the reference-loop correctness harness
research_context.py     builds the panel once; owns the dev/holdout split
strategies/             baseline_strategy.py, alpha_01..06, _fitting.py
```

`research_context.py`, `portfolio_backtest.py` and `plots.py` are additions beyond
the mandated list, permitted by §1.3. The first exists so that every downstream
script draws its data from one place and two tables in the same report cannot
disagree; the reasoning for the other two is in the Task 3 report.

## 7. Stated assumptions

1. **Shorting is permitted.** Nothing in the problem statement forbids it and the
   dominant effect in this instrument is two-sided. Positions are bounded to [−1, +1].
2. **Positions are a fraction of equity**, not a share count.
3. **Risk-free rate is zero.** Sharpe and Sortino are computed on raw returns; with
   a single instrument and no financing detail supplied, any other choice would be
   invented.
4. **The terminal candle is forced flat** and charged its exit, since no forward
   open exists to hold it into.
5. **Headline slippage is zero** beyond the mandated cost, for the comparability
   reason given in §2.
6. **`statistics.py` shadows the standard library.** The filename is mandated, and
   it does genuinely shadow Python's own `statistics` module for anything imported
   afterwards. Rather than break the requirement or leave the hazard in place, the
   stdlib module is loaded from its own path and its public names are re-exported,
   so `import statistics` keeps working for third-party code either way.

## 8. Reproducibility

All randomness is seeded from `config.RANDOM_SEED`. Paths resolve relative to
`config.py`, so the project runs from any working directory with no edits. Every
printed table is also written to `results/`, and the full console output to
`results/run_log.txt`.
