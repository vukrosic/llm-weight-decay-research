import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_DIR = Path('experiments/01_muon_weight_decay_focus/runs/muon_kimi_update_local_search')


def parse_wd(name: str) -> float:
    tok = name.split('_')[1].replace('wd', '')
    return float(tok.replace('m', '-').replace('p', '.'))


def main():
    rows = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / 'metrics.json'
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        fm = data.get('final_metrics', {})
        rows.append(
            {
                'run': d.name,
                'wd': parse_wd(d.name),
                'val_loss': float(fm.get('val_loss')),
                'train_loss': float(fm.get('train_loss')),
                'val_acc': float(fm.get('val_accuracy')),
            }
        )

    if not rows:
        raise SystemExit('No completed runs found.')

    baseline = next((r for r in rows if abs(r['wd']) < 1e-12), None)
    for r in rows:
        r['delta_vs_wd0'] = r['val_loss'] - baseline['val_loss'] if baseline else float('nan')

    rows_sorted = sorted(rows, key=lambda x: x['val_loss'])
    (RUNS_DIR / 'summary.json').write_text(json.dumps(rows_sorted, indent=2))

    xs = [r['wd'] for r in sorted(rows, key=lambda x: x['wd'])]
    ys = [r['val_loss'] for r in sorted(rows, key=lambda x: x['wd'])]
    plt.figure(figsize=(10, 5.5))
    plt.plot(xs, ys, marker='o', linewidth=1.8)
    plt.xlabel('Muon update-mode wd')
    plt.ylabel('Final val loss')
    plt.title('Muon update-mode local search')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    p1 = RUNS_DIR / 'val_loss_vs_wd.png'
    plt.savefig(p1, dpi=220)
    plt.close()

    deltas = [r['delta_vs_wd0'] for r in sorted(rows, key=lambda x: x['wd'])]
    plt.figure(figsize=(10, 5.5))
    colors = ['#16a34a' if d < 0 else '#dc2626' for d in deltas]
    plt.bar(xs, deltas, width=0.03, color=colors)
    plt.axhline(0.0, color='black', linewidth=1)
    plt.xlabel('Muon update-mode wd')
    plt.ylabel('Delta vs wd=0')
    plt.title('Update-mode impact vs baseline')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    p2 = RUNS_DIR / 'delta_vs_wd0.png'
    plt.savefig(p2, dpi=220)
    plt.close()

    best = rows_sorted[0]
    report = RUNS_DIR / 'REPORT.md'
    with report.open('w') as f:
      f.write('# Muon Update-Mode Local Search Report (Single Seed)\n\n')
      f.write('wd | val_loss | train_loss | val_acc | delta_vs_wd0\n')
      f.write('---|---:|---:|---:|---:\n')
      for r in rows_sorted:
          f.write(f"{r['wd']} | {r['val_loss']:.4f} | {r['train_loss']:.4f} | {r['val_acc']:.4f} | {r['delta_vs_wd0']:+.4f}\n")
      f.write('\n')
      f.write(f"Best wd: `{best['wd']}` with val_loss `{best['val_loss']:.4f}`\n")
      if baseline:
          f.write(f"Baseline wd=0 val_loss: `{baseline['val_loss']:.4f}`\n")
          f.write(f"Best delta vs wd=0: `{best['delta_vs_wd0']:+.4f}`\n")
      f.write('\n![Val loss vs wd](./val_loss_vs_wd.png)\n')
      f.write('\n![Delta vs wd0](./delta_vs_wd0.png)\n')

    print(f'Wrote: {report}')
    print(f'Wrote: {p1}')
    print(f'Wrote: {p2}')


if __name__ == '__main__':
    main()
