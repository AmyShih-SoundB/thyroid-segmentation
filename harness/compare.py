"""Print the W4 baseline ladder from per-case score.py CSVs.

Usage:
  python compare.py "U-Net=/path/notes/15_score_run008_predict_fold*.csv" \
                    "MTNet=/path/notes/14_score_run009_fold*.csv" ...
The FIRST model given is the baseline for the paired comparison.
"""
import csv, glob, statistics as st, sys

METRICS = ('dice', 'iou', 'hd95')

def fold_means(pattern):
    """-> list of {metric: mean} per fold, ordered by filename."""
    out = []
    for path in sorted(glob.glob(pattern)):
        rows = list(csv.DictReader(open(path)))
        m = {}
        for k in METRICS:
            vals = [float(r[k]) for r in rows if r[k] not in ('', None)]
            m[k] = st.mean(vals) if vals else float('nan')
        m['n'] = len(rows)
        out.append(m)
    if not out:
        sys.exit(f"no CSVs matched: {pattern}")
    return out

def msd(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)

specs = [a.split('=', 1) for a in sys.argv[1:]]
if not specs:
    sys.exit(__doc__)
models = [(label, fold_means(pat)) for label, pat in specs]

for k in METRICS:
    print(f"\n=== {k} ===")
    print(f"{'model':<12}" + "".join(f"{'f'+str(i):>10}" for i in range(len(models[0][1])))
          + f"{'mean':>10}{'SD':>9}")
    for label, folds in models:
        vals = [f[k] for f in folds]
        mean, sd = msd(vals)
        print(f"{label:<12}" + "".join(f"{v:10.4f}" for v in vals)
              + f"{mean:10.4f}{sd:9.4f}")

    base_label, base = models[0]
    for label, folds in models[1:]:
        if len(folds) != len(base):
            print(f"  {label}: fold count differs from {base_label}, paired test skipped")
            continue
        d = [a[k] - b[k] for a, b in zip(folds, base)]
        mean, sd = msd(d)
        se = sd / len(d) ** 0.5 if sd else 0.0
        t = mean / se if se else float('inf')
        wins = sum(1 for x in d if (x < 0 if k == "hd95" else x > 0))
        print(f"  {label} - {base_label}: mean {mean:+.4f}  SD {sd:.4f}  "
              f"SE {se:.4f}  t {t:+.2f} ({len(d)-1} df)  folds improved {wins}/{len(d)}")

print(f"\ncases per fold: " + ", ".join(
    f"{label} {folds[0]['n']}" for label, folds in models))
print("SD is across folds, not across cases. Paired test is fold-wise.")
