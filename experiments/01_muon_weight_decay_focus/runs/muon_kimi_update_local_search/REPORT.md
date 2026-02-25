# Muon Update-Mode Local Search Report (Single Seed)

wd | val_loss | train_loss | val_acc | delta_vs_wd0
---|---:|---:|---:|---:
-0.4 | 7.2774 | 7.3013 | 0.1304 | -0.0142
-0.25 | 7.2803 | 7.2949 | 0.1292 | -0.0113
-0.2 | 7.2826 | 7.2911 | 0.1299 | -0.0090
-0.15 | 7.2830 | 7.2920 | 0.1303 | -0.0087
-0.1 | 7.2844 | 7.2937 | 0.1298 | -0.0072
-0.3 | 7.2885 | 7.2891 | 0.1282 | -0.0031
-0.05 | 7.2885 | 7.2987 | 0.1293 | -0.0031
0.0 | 7.2916 | 7.3042 | 0.1289 | +0.0000

Best wd: `-0.4` with val_loss `7.2774`
Baseline wd=0 val_loss: `7.2916`
Best delta vs wd=0: `-0.0142`

![Val loss vs wd](./val_loss_vs_wd.png)

![Delta vs wd0](./delta_vs_wd0.png)
