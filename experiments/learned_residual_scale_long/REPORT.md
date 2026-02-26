# Learned Residual Scale Long-Run Report (Single Seed)

variant | mode | init | val_loss | train_loss | val_acc | attn_scale_mean | ff_scale_mean | layer_scale_mean | delta_vs_baseline
---|---|---:|---:|---:|---:|---:|---:|---:|---:
learned_branch:init0.1 | learned_branch | 0.1 | 4.1261 | 4.4661 | 0.3168 | 0.0963 | 0.0154 |  | -0.0639
learned_layer:init0.1 | learned_layer | 0.1 | 4.1459 | 4.4902 | 0.3143 |  |  | 0.0804 | -0.0441
fixed:1 | fixed | 1.0 | 4.1900 | 4.5394 | 0.3082 |  |  |  | +0.0000
learned_branch:init0.01 | learned_branch | 0.01 | 4.6005 | 4.8951 | 0.2944 | 0.0196 | 0.0021 |  | +0.4105

Best variant: `learned_branch:init0.1` with val_loss=`4.1261`
Baseline (fixed:1.0) val_loss=`4.1900`
Best delta vs baseline: `-0.0639`

![Ranked val loss](./ranked_val_loss.png)

![Delta vs baseline](./delta_vs_baseline.png)

![All loss lines](./all_loss_lines.png)
