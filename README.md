# Multi-Alpha Research Lab

Inter-IIT Prepathon, Quantitative Finance. This folder is the Task 3 submission and
contains the whole project (Tasks 1, 2 and 3), since each task builds on the last.

---

## AI use disclosure

I used AI tools to help refine the reports and code. My purpose was to learn from
the suggestions and strengthen my understanding of the methods and implementation,
not to just vibe code and submit.


---

## Running it

```bash
pip install -r requirements.txt
python main.py
```

That runs all three tasks, writes every table to `results/`, every figure to
`results/figures/` and the full console output to `results/run_log.txt`. It takes about
90 seconds on a laptop.

| Command | What it does |
|---|---|
| `python main.py` | everything |
| `python main.py --task 1` | data pipeline, integrity checks, baseline |
| `python main.py --task 2` | alpha research: statistics, robustness, orthogonality, selection |
| `python main.py --task 3` | Task 2 first (Task 3 needs its selection), then allocation |
| `python main.py --quick` | fewer bootstrap and permutation draws |

Paths are resolved relative to `config.py`, so it runs from any directory without edits.
Results are deterministic (fixed seeds), and the `results/` folder shipped here is exactly
what the command above produces.

---

## Reports

| | |
|---|---|
| [`reports/implementation_note.md`](reports/implementation_note.md) | Task 1: design, the data defects we found, integrity checks |
| [`reports/task2_research_report.md`](reports/task2_research_report.md) | Task 2: six hypotheses, evidence, orthogonality, selection |
| [`reports/task3_research_report.md`](reports/task3_research_report.md) | Task 3: factor model, allocation comparison, the learned component and its null tests |

---

## What we found

The instrument mean-reverts at short horizons, so the trend flags in the signal library
work as contrarian indicators. The naive baseline that buys when PB01 is on loses 3.0% a
year in an asset that rose 36%.

Task 2 built six strategies and selected three (alpha_02, alpha_03, alpha_05) by a rule
written before the holdout was opened. The six behave like about two independent streams
(participation ratio 1.98). alpha_05 was the best on development (Sharpe 1.02) and the
worst on the holdout (−1.45). We kept it in the book.

Task 3 allocates across those three plus alpha_07, a strategy added after the Task 2
holdout had been seen (the report explains why, and why its holdout numbers should be
discounted). On the 2021 holdout:

- equal weight made +4.9%/yr at Sharpe 1.59 and a −2.4% max drawdown, against +2.0% and
  −5.8% for buy and hold. Without alpha_07 the same book made +2.8% at Sharpe 0.95.
- the learned allocation model does not beat its shuffled-label null (p = 0.24) or a
  random-tilt null (p = 0.18 on dev, 0.87 on the holdout). The allocator is gated so it
  can't act on a model that hasn't shown skill, and the gate never opened in the holdout.
- our recommendation is equal weight, with risk parity as the alternative if drawdown
  matters more than return.

---

## Layout

```
main.py                   the one entry point
config.py                 paths, mandated cost, dev/holdout boundary

  Task 1: infrastructure
data_loader.py            load and schema-check
data_cleaner.py           repairs and data-quality report
feature_engine.py         decision frame; owns the t-1 information cutoff
strategy.py               BaseStrategy, the interface every alpha implements
execution_engine.py       fills at the open, costs, slippage, trade log
portfolio.py              positions, equity, P&L
performance.py            every metric, for strategies and allocations alike
backtester.py             the loop, plus a check against a literal per-candle loop

  Task 2: research
alpha_research.py         the Task 2 pipeline and the selection rule
statistics.py             significance tests, multiple-testing correction
robustness.py             sub-periods, regimes, costs, parameters, noise, failures
orthogonality.py          QR / return-space analysis

  Task 3: portfolio
factor_model.py           alpha vs systematic exposure
portfolio_optimizer.py    equal weight, inverse vol, risk parity, max Sharpe, MV, risk shares
meta_model.py             learned effectiveness model, walk-forward, permuted-label null
dynamic_allocator.py      skill-gated tilting, plus the random-tilt null
final_evaluation.py       the required head-to-head comparison

  supporting (allowed by Tech Doc §1.3)
research_context.py       builds the data once and owns the split
portfolio_backtest.py     nets strategy positions into one book before costing
plots.py                  figures

strategies/               baseline_strategy.py, alpha_01..alpha_07, _fitting.py
data/                     price_train.csv, signals_train.csv
results/                  generated tables, figures and run log
```

`strategies/__init__.py` keeps two registries. `ALPHA_REGISTRY` is the Task 2 candidate
set exactly as submitted (alpha_01 to alpha_06). `TASK3_ADDITIONS` holds alpha_07, so
adding it didn't change the Task 2 run. All 48 Task 1 and Task 2 output tables match the
18 Sep submission.

---

## Execution assumptions

Fixed by the Tech Doc and enforced in code:

- 0.05% of traded notional per side, so a round trip costs 0.10%
- the decision for candle t uses signal values up to t−1's close only
- trades fill at candle t's open
- price and volume are never strategy inputs; they are used only as the training label
  and for execution and accounting
- headline slippage is zero beyond the mandated cost; slippage from 0 to 10 bps per side
  is tested in the cost-sensitivity study (reasoning in the Task 1 note, §2)

Each of these is checked on every run and raises an error if it fails. The vectorised
engine matches a literal per-candle loop to 6.7 × 10⁻¹⁶.

Allocations are costed on the netted book (sum of weight × position), not by adding up
each strategy's pre-costed returns, so offsetting trades between strategies aren't
charged twice.

---

## A note on `statistics.py`

The Tech Doc requires this filename, and it shadows Python's standard-library
`statistics` module. To keep the required name without breaking anything else, the
module loads the standard-library version from its own path and re-exports its public
names, so `import statistics` still works for other code.
