"""CPU/stdlib-only analysis of legacy epoch logs and clip_steps_ep*.jsonl.

Example: python3 scripts/analysis/analyze_clip_grad.py --exp outputs/0025_* \
    --exp outputs/0026_* --exp outputs/0028_* --out docs/analysis/clip_grad
Legacy global epoch means NEVER determine per-parameter clipping frequency.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values), median=quantile(values, .5), p90=quantile(values, .9),
                p95=quantile(values, .95), max=max(values) if values else None)


def summarize_steps(rows):
    valid = [r for r in rows if r['finite']]
    n = len(valid)
    params = sum(r['n_params'] for r in valid)
    return dict(
        steps=len(rows), nonfinite_steps=len(rows) - n,
        step_norm=stats([r['pre_clip_norm'] for r in valid]),
        clipped_step_fraction=sum(r['n_clipped'] > 0 for r in valid) / n if n else None,
        above_threshold_step_fraction=sum(r['n_above_threshold'] > 0 for r in valid) / n if n else None,
        clipped_parameter_fraction=sum(r['n_clipped'] for r in valid) / params if params else None,
        coef_min=stats([r['coef_min'] for r in valid]),
        coef_parameter_weighted_mean=sum(r['coef_mean'] * r['n_params'] for r in valid) / params if params else None,
    )


def analyze(directory):
    config = json.loads((directory / 'config.json').read_text())
    by_epoch = {}
    sources = []
    duplicates = 0
    # Sort by full timestamp from content, NOT HHMMSS filenames across dates.
    logs = []
    for path in directory.glob('log_*.txt'):
        if re.search(r'_rank\d+\.txt$', path.name):
            continue
        for line in path.read_text().splitlines():
            match = re.search(r'\[(\d{8}-\d{6})\].*Epoch: (\d+), (.*)', line)
            if match:
                fields = dict(re.findall(r'(\w+): ([^,\s]+)', match[3]))
                if 'grad_norm' in fields:
                    logs.append((match[1], path.name, int(match[2]), fields))
        sources.append(str(path))
    for timestamp, source, epoch, fields in sorted(logs):
        duplicates += epoch in by_epoch
        by_epoch[epoch] = dict(epoch=epoch, timestamp=timestamp, source=source,
                               **{k: float(fields[k]) for k in ('grad_norm', 'train_loss', 'lr')})
    records = [by_epoch[e] for e in sorted(by_epoch)]
    if not records:
        raise ValueError(f'No epoch gradient logs: {directory}')
    # Atomic epoch files; choose newest attempt and do not double count resumes.
    telemetry = {}
    telemetry_sources = {}
    for path in sorted(directory.glob('clip_steps_ep*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            continue
        epoch = rows[0]['epoch']
        if any(r['epoch'] != epoch for r in rows) or [r['step'] for r in rows] != list(range(len(rows))):
            raise ValueError(f'Invalid step sequence: {path}')
        telemetry[epoch] = rows
        telemetry_sources[epoch] = str(path)
    last = max(by_epoch)
    phases = []
    # Observed thirds describe each run's lifetime, not matched training ages.
    for label, lo, hi in [('early', 1, last // 3), ('middle', last // 3 + 1, 2 * last // 3),
                           ('late', 2 * last // 3 + 1, last)]:
        subset = [r for r in records if lo <= r['epoch'] <= hi]
        steps = [r for e, rows in telemetry.items() if lo <= e <= hi for r in rows]
        phases.append(dict(phase=label, start=lo, end=hi,
                           epoch_mean_norm=stats([r['grad_norm'] for r in subset]),
                           clipping=summarize_steps(steps) if steps else None))
    return dict(name=directory.name, threshold=config['clip_grad'],
                configured_epochs=config.get('num_epoch'), sources=sources,
                duplicate_epoch_rows=duplicates, epochs=records,
                missing_epochs=sorted(set(range(1, last + 1)) - set(by_epoch)),
                epoch_mean_norm=stats([r['grad_norm'] for r in records]),
                phases=phases, telemetry_sources=telemetry_sources,
                clipping_by_epoch={e: summarize_steps(rows) for e, rows in sorted(telemetry.items())},
                clipping=summarize_steps([r for rows in telemetry.values() for r in rows]) if telemetry else None)


def report(experiments):
    lines = ['# clip_grad log analysis', '',
             'Norm statistics below describe **epoch means of global pre-clip L2 norms**, not step quantiles.',
             'Clipping is per parameter. Missing step telemetry is unknown, not zero.', '',
             '| Experiment | Threshold | Epochs | Median | p90 | p95 | Max | Clipped steps |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for exp in experiments:
        s = exp['epoch_mean_norm']
        clip = exp['clipping']
        frequency = str(clip['clipped_step_fraction']) if clip else 'unknown'
        lines.append(f"| {exp['name']} | {exp['threshold']} | {s['n']} | {s['median']:.4f} | {s['p90']:.4f} | {s['p95']:.4f} | {s['max']:.4f} | {frequency} |")
    lines += ['', '## Observed lifetime thirds', '',
              'Different epoch ranges across runs; these are not matched-age comparisons.', '',
              '| Experiment | Phase | Epoch range | Median epoch mean | p95 epoch mean | Clipped steps |',
              '|---|---|---|---:|---:|---|']
    for exp in experiments:
        for phase in exp['phases']:
            s = phase['epoch_mean_norm']
            if not s['n']:
                continue
            clip = phase['clipping']
            frequency = str(clip['clipped_step_fraction']) if clip else 'unknown'
            lines.append(f"| {exp['name'][:4]} | {phase['phase']} | {phase['start']}–{phase['end']} | {s['median']:.4f} | {s['p95']:.4f} | {frequency} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Output filename prefix')
    args = parser.parse_args()
    experiments = [analyze(path) for path in args.exp]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(experiments, indent=2, allow_nan=False) + '\n')
    args.out.with_suffix('.md').write_text(report(experiments))
    with args.out.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['experiment', 'epoch', 'timestamp', 'source', 'grad_norm', 'train_loss', 'lr'])
        writer.writeheader()
        for exp in experiments:
            writer.writerows(dict(experiment=exp['name'], **row) for row in exp['epochs'])
    print(report(experiments))


if __name__ == '__main__':
    main()
