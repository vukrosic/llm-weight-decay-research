import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_DIR = Path('experiments/01_muon_weight_decay_focus/runs/muon_kimi_idea_search')


def main():
    rows = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / 'metrics.json'
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        ec = data.get('experiment_config', {})
        fm = data.get('final_metrics', {})
        rows.append(
            {
                'run': d.name,
                'mode': ec.get('muon_decay_mode', 'param'),
                'wd': float(ec.get('muon_weight_decay', 0.0)),
                'val_loss': float(fm.get('val_loss')),
                'train_loss': float(fm.get('train_loss')),
                'val_acc': float(fm.get('val_accuracy')),
            }
        )

    if not rows:
        raise SystemExit('No completed runs found.')

    baseline = next((r for r in rows if r['mode'] == 'param' and abs(r['wd']) < 1e-12), None)
    if baseline:
        for r in rows:
            r['delta_vs_baseline'] = r['val_loss'] - baseline['val_loss']
    else:
        for r in rows:
            r['delta_vs_baseline'] = float('nan')

    rows_sorted = sorted(rows, key=lambda x: x['val_loss'])
    (RUNS_DIR / 'summary.json').write_text(json.dumps(rows_sorted, indent=2))

    labels = [f"{r['mode']}:{r['wd']:g}" for r in rows_sorted]
    vals = [r['val_loss'] for r in rows_sorted]
    deltas = [r['delta_vs_baseline'] for r in rows_sorted]

    plt.figure(figsize=(max(10, len(rows_sorted) * 0.55), 5.5))
    plt.bar(range(len(rows_sorted)), vals, color='#0ea5e9')
    plt.xticks(range(len(rows_sorted)), labels, rotation=55, ha='right')
    plt.ylabel('Final validation loss')
    plt.title('Muon idea search: ranked by val loss')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    p1 = RUNS_DIR / 'ranked_val_loss.png'
    plt.savefig(p1, dpi=220)
    plt.close()

    plt.figure(figsize=(max(10, len(rows_sorted) * 0.55), 5.5))
    colors = ['#16a34a' if d < 0 else '#dc2626' for d in deltas]
    plt.bar(range(len(rows_sorted)), deltas, color=colors)
    plt.axhline(0.0, color='black', linewidth=1)
    plt.xticks(range(len(rows_sorted)), labels, rotation=55, ha='right')
    plt.ylabel('Val loss delta vs baseline')
    plt.title('Muon idea search: impact vs param wd=0 baseline')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    p2 = RUNS_DIR / 'delta_vs_baseline.png'
    plt.savefig(p2, dpi=220)
    plt.close()

    best = rows_sorted[0]
    report = RUNS_DIR / 'REPORT.md'
    with report.open('w') as f:
        f.write('# Muon Kimi-Style Idea Search Report (Single Seed)\n\n')
        f.write('mode | wd | val_loss | train_loss | val_acc | delta_vs_baseline\n')
        f.write('---|---:|---:|---:|---:|---:\n')
        for r in rows_sorted:
            f.write(
                f"{r['mode']} | {r['wd']} | {r['val_loss']:.4f} | {r['train_loss']:.4f} | "
                f"{r['val_acc']:.4f} | {r['delta_vs_baseline']:+.4f}\n"
            )
        f.write('\n')
        f.write(f"Best idea: mode=`{best['mode']}`, wd=`{best['wd']}` with val_loss=`{best['val_loss']:.4f}`\n")
        if baseline:
            f.write(f"Baseline (param, wd=0): `{baseline['val_loss']:.4f}`\n")
            f.write(f"Best delta vs baseline: `{best['delta_vs_baseline']:+.4f}`\n")
        f.write('\n![Ranked val loss](./ranked_val_loss.png)\n')
        f.write('\n![Delta vs baseline](./delta_vs_baseline.png)\n')

    print(f'Wrote: {report}')
    print(f'Wrote: {p1}')
    print(f'Wrote: {p2}')


if __name__ == '__main__':
    main()
