# Learned Residual Scale 100M Pair Report (Single Seed)

variant | val_loss | train_loss | val_acc | delta_vs_baseline | attn_scale_mean | ff_scale_mean
---|---:|---:|---:|---:|---:|---:
fixed:1 | 3.7474 | 3.7727 | 0.3486 | +0.0000 |  | 
learned_branch:init0.1 | 3.7992 | 3.8287 | 0.3474 | +0.0518 | 0.0957 | 0.0106

![Final val compare](./final_val_loss_compare.png)

![All loss lines](./all_loss_lines.png)
