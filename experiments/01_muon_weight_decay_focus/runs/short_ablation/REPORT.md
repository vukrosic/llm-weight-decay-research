# Short Regularization Ablation Report

## Ranking by Final Validation Loss

1. `smallwd_001_lr02_adamw02_seed42` | val_loss=7.2822 | train_loss=7.2908 | val_acc=0.1304 | muon_wd=0.01 | muon_lr=0.02 | adamw_wd=None
2. `smallwd_001_lr016_adamw02_seed42` | val_loss=7.2870 | train_loss=7.2868 | val_acc=0.1285 | muon_wd=0.01 | muon_lr=0.016 | adamw_wd=None
3. `negwd_001_lr024_adamw02_seed42` | val_loss=7.2874 | train_loss=7.2982 | val_acc=0.1297 | muon_wd=-0.01 | muon_lr=0.024 | adamw_wd=None
4. `negwd_0005_lr024_adamw02_seed42` | val_loss=7.2890 | train_loss=7.2996 | val_acc=0.1294 | muon_wd=-0.005 | muon_lr=0.024 | adamw_wd=None
5. `refwd_01_lr024_adamw02_seed42` | val_loss=7.2900 | train_loss=7.3005 | val_acc=0.1294 | muon_wd=0.1 | muon_lr=0.024 | adamw_wd=None
6. `baseline_wd0_lr024_adamw02_seed42` | val_loss=7.2904 | train_loss=7.3023 | val_acc=0.1294 | muon_wd=0.0 | muon_lr=0.024 | adamw_wd=None
7. `smallwd_0005_lr024_adamw02_seed42` | val_loss=7.2905 | train_loss=7.3026 | val_acc=0.1292 | muon_wd=0.005 | muon_lr=0.024 | adamw_wd=None
8. `refwd_005_lr024_adamw02_seed42` | val_loss=7.2913 | train_loss=7.3036 | val_acc=0.1289 | muon_wd=0.05 | muon_lr=0.024 | adamw_wd=None
9. `smallwd_0001_lr024_adamw02_seed42` | val_loss=7.2915 | train_loss=7.3035 | val_acc=0.1288 | muon_wd=0.001 | muon_lr=0.024 | adamw_wd=None
10. `smallwd_001_lr024_adamw00_seed42` | val_loss=7.2916 | train_loss=7.3034 | val_acc=0.1291 | muon_wd=0.01 | muon_lr=0.024 | adamw_wd=None
11. `smallwd_001_lr024_adamw02_seed42` | val_loss=7.2916 | train_loss=7.3041 | val_acc=0.1289 | muon_wd=0.01 | muon_lr=0.024 | adamw_wd=None
12. `negwd_0001_lr024_adamw02_seed42` | val_loss=7.2925 | train_loss=7.3048 | val_acc=0.1286 | muon_wd=-0.001 | muon_lr=0.024 | adamw_wd=None
13. `smallwd_001_lr028_adamw02_seed42` | val_loss=7.2970 | train_loss=7.3053 | val_acc=0.1278 | muon_wd=0.01 | muon_lr=0.028 | adamw_wd=None

## Plot

![Val Loss Ranking](./val_loss_ranking.png)

![Val Loss Delta to Best](./val_loss_delta_to_best.png)
